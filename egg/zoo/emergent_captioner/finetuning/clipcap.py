# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList
import os
from egg.zoo.emergent_captioner.finetuning.utils import (
    KLRegularizer,
    StopTokenLogitsProcessor,
)
from egg.zoo.emergent_captioner.utils import convert_models_to_fp32
from egg.zoo.emergent_captioner.finetuning.utils import int2mil, trainable_params
import time
from torch.cuda.amp import autocast
from contextlib import nullcontext
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from PIL import Image

def cocoid2img(cocoid):

    if os.path.isdir("/workspace/manugaur"):
        img_path = f"/workspace/manugaur/coco/train2014/COCO_train2014_{int(cocoid):012d}.jpg"
        if not os.path.isfile(img_path):
            img_path = f"/workspace/manugaur/coco/val2014/COCO_val2014_{int(cocoid):012d}.jpg"
    else:
        img_path = f"/home/ubuntu/pranav/pick_edit/coco/train2014/COCO_train2014_{int(cocoid):012d}.jpg"
        if not os.path.isfile(img_path):
            img_path = f"/home/ubuntu/pranav/pick_edit/coco//val2014/COCO_val2014_{int(cocoid):012d}.jpg"
        
    return img_path

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MlpTransformer(nn.Module):
    def __init__(
        self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.0
    ):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim_self,
        dim_ref,
        num_heads,
        mlp_ratio=4.0,
        bias=False,
        dropout=0.0,
        act=F.relu,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(
            dim_self, dim_ref, num_heads, bias=bias, dropout=dropout
        )
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(
            dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout
        )

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim_self: int,
        num_heads: int,
        num_layers: int,
        dim_ref: Optional[int] = None,
        mlp_ratio: float = 2.0,
        act=F.relu,
        norm_layer: nn.Module = nn.LayerNorm,
        enc_dec: bool = False,
    ):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_self,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            else:  # self or cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):
    def __init__(
        self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        clip_length: int,
        num_layers: int = 8,
    ):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )

    """output learnable tokens from transformer. They will learn to exctract visual information required for captioning. Kinda like cross-attention"""
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length :]
        return out


class ClipCapModel(nn.Module):
    def __init__(
        self,
        clip_prefix_size: int,
        nb_prefix_tokens: int = 10,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapModel, self).__init__()

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.do_sample = do_sample
        self.beam_size = beam_size
        self.max_len = max_len

        self.logits_processor = StopTokenLogitsProcessor(self.tokenizer, do_sample)

        gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        input_dim = clip_prefix_size

        self.nb_prefix_tokens = nb_prefix_tokens

        hidden_dim = (gpt_embedding_size * self.nb_prefix_tokens) // 2
        output_dim = gpt_embedding_size * self.nb_prefix_tokens
        self.clip_project = MLP((input_dim, hidden_dim, output_dim))

        self.kl_regularizer = KLRegularizer()

    def forward(self, image_feats, aux_input=None, CIDER_OPTIM = False, use_fp16= False, greedy_baseline = False, train_method = None):
        if train_method == "mle" and self.training: 
            prompts = self.clip_project(image_feats) #16,7680
            prompts = prompts.view(image_feats.shape[0], self.nb_prefix_tokens, -1) # 16,10,768

            bsz, prefix_len, h_dim = prompts.shape
            tokens_flat = aux_input['tokens'].view(-1,aux_input['tokens'].shape[-1]) # B*5, 40
            token_emb = self.gpt.transformer.wte(tokens_flat) #B*5, 40 , 768
            gpt_input = torch.cat((prompts, token_emb), dim = 1) # B*5, 50, 768
            mask = aux_input['mask'].view(-1, aux_input['mask'].shape[-1]) # B*5, 50
            out = self.gpt(inputs_embeds = gpt_input, attention_mask = mask)
            return out
        
        else:
            prompts = self.clip_project(image_feats) #16,7680
            prompts = prompts.view(image_feats.shape[0], self.nb_prefix_tokens, -1) # 16,10,768

            bsz, prefix_len, h_dim = prompts.shape

            prompts_flat = prompts.view(-1, prompts.size(-1)) #B*10,768
            start = len(self.tokenizer) #50257
            end = start + (bsz * prefix_len) # 50417 as prefix len = 10 and bsz = 16
            input_ids = torch.arange(start, end).view(*prompts.shape[:2]).to(prompts.device)
            self.gpt.get_input_embeddings().weight.data[start:end] = prompts_flat #add prefix tokens for batch images to GPT lookup table

            if not greedy_baseline and self.training:
                self.do_sample = True
                temp = 0.3
            else:
                temp = 1.0
            flag = False

            if self.training or greedy_baseline:
                generated = self.gpt.generate(
                    input_ids,
                    do_sample=self.do_sample,
                    max_length=self.max_len,
                    num_beams=self.beam_size,
                    num_return_sequences=1,
                    logits_processor=LogitsProcessorList([self.logits_processor]),
                    top_k= len(self.tokenizer),
                    temperature = temp
                )
                
            else:
                # at test time we use beam search regardless of the decoding method used at training time
                print("BEAM SEARCH GENERATION")
                generated = self.gpt.generate(
                    input_ids,
                    do_sample=False,
                    max_length=self.max_len,
                    num_beams=5,
                    num_return_sequences=1,
                    logits_processor=LogitsProcessorList([self.logits_processor]),
                    top_k=len(self.tokenizer),
                )

            indices = generated[:, prefix_len:] # generated (B, max_batch_len) tokens. After "."/13 padded with "<eos>/50256

            # logits after generation to get logprobs
            suffix = self.gpt.get_input_embeddings()(indices) # B, 10, 768 embd. 
            inputs_embeds = torch.cat([prompts, suffix], dim=1) # B, 20,768 emb

            logits = self.gpt(inputs_embeds=inputs_embeds)
            logits = logits[0][:, prefix_len - 1 : -1, : len(self.tokenizer)] # extract last logit from --> logits[0] : (B, max_batch_len, 55257) 
            logits = logits/(temp if temp > 0 else 1.0)
            logits = logits.log_softmax(-1)

            # compute_mask and msg_lengths
            max_k = indices.size(1) #len of longest generation in batch i.e 10
            end_of_caption = indices == self.eos_token_id #50256 padding tokens
            extra_tokens = end_of_caption.cumsum(dim=1) > 0
            msg_lengths = max_k - (extra_tokens).sum(dim=1)
            # msg_lengths.add_(1).clamp_(max=max_k) #lengths increased by 1?

            mask = (extra_tokens == 0).float()

            # get logprob for each sampled token for all captions

            log_probs = torch.gather(logits, dim=2, index=indices.unsqueeze(2)).squeeze(-1) # (B, 10) : log prob for each sampled policy word
            log_probs *= mask # everything after "." is zeroed
            log_probs = log_probs.sum(1) / msg_lengths #averaged

            decoded_captions = self.tokenizer.batch_decode(
                indices,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            self.do_sample = False

        return decoded_captions, log_probs, torch.randn(1)

    def maybe_patch_gpt(self, max_embeddings):
        if not getattr(self.gpt, "_patched", False):
            self.gpt._patched = True
            #increase gpt nn.embedding from vocab size --> + 5000
            self.gpt.resize_token_embeddings(len(self.tokenizer) + max_embeddings)
            
            # bias for output embeddings (768) added (55257)
            if self.gpt.get_output_embeddings().bias is None:
                self.gpt.get_output_embeddings().bias = torch.nn.Parameter(
                    torch.tensor([0.0] * (len(self.tokenizer) + max_embeddings))
                )
                self.gpt.get_output_embeddings().bias.requires_grad = False
                self.gpt.get_output_embeddings().to(
                    self.gpt.get_output_embeddings().weight.device
                )
                self.gpt._originally_with_no_bias = True
            else:
                self.gpt._originally_with_no_bias = False
            self.gpt.get_output_embeddings().bias.data[-max_embeddings:] = float("-inf")#-1e+4

    def maybe_unpatch_gpt(self):
        if getattr(self.gpt, "_patched", False):
            self.gpt._patched = False
            self.gpt.resize_token_embeddings(len(self.tokenizer))
            if self.gpt._originally_with_no_bias:
                self.gpt.get_output_embeddings().bias = None


class ClipCapSender(nn.Module):
    def __init__(
        self,
        clip_model: str,
        clipcap_path: str,
        official_clipcap_weights : str,
        train_method : str,
        config : dict,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapSender, self).__init__()

        assert max_len < 75  # clip maximum context size

        # self.clip_vit = clip.load(clip_model)[0].visual
        self.finetune_llm = config["finetune_model"]=="llm"
        self.clip, self.clip_preproc = clip.load(clip_model)
        convert_models_to_fp32(self.clip)

        # self.clip.eval()
        """
        if gpt finetuning :  freeze clip. 
                             lora decides what to freeze in sender.clipcap
        if clip finetuning : gpt.transformer frozen in LoRA only. 
                             won't be finetuning CLIP w/o LoRA anyways.
        """
        if self.finetune_llm:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.clipcap = ClipCapModel(
            clip_prefix_size=self.clip.visual.output_dim,
            do_sample=do_sample,
            beam_size=beam_size,
            max_len=max_len,
        )
        if train_method != "mle":
            if clipcap_path is not None:
                print(f"| LOADED MODEL : {clipcap_path}")

                trained_wts = torch.load(clipcap_path)
                model_params = self.state_dict()
                for k in model_params.keys():
                    if k in trained_wts:
                        model_params[k] = trained_wts[k]
                
                self.load_state_dict(model_params)


    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None, CIDER_OPTIM= False, use_fp16 = False, greedy_baseline = False, train_method = None):
        #if loading CLIP on GPU
        image_feats = self.clip.visual(images)
        
        if train_method == "mle":
            if self.training:
                image_feats = self.repeat_tensors(aux_input['tokens'].shape[1], image_feats) #ABC --> AAA..BBB..CCC..
            return self.clipcap(image_feats, aux_input, CIDER_OPTIM, use_fp16,greedy_baseline,train_method)

        else:
            captions, log_probs, kl_div = self.clipcap(image_feats, aux_input, use_fp16, CIDER_OPTIM, greedy_baseline, train_method)
            return captions, log_probs, kl_div

    def repeat_tensors(self, n, x):
        """
        For a tensor of size Bx..., we repeat it n times, and make it Bnx...
        For collections, do nested repeat
        """
        if torch.is_tensor(x):
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:]))) # Bxnx...
            x = x.reshape(x.shape[0]*n, *x.shape[2:]) # Bnx...
        elif type(x) is list or type(x) is tuple:
            x = [repeat_tensors(n, _) for _ in x]
        return x

    # def named_parameters(self, prefix="", recurse: bool = True):
    #     return self.clipcap.named_parameters()

    # def parameters(self, recurse: bool = True):
    #     if self.finetune_llm:
    #         return self.clipcap.parameters()
    #     else:
    #         return self.parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.clipcap.train(mode)
        return self

    def patch_model(self, batch_size: int = 500, prefix_len: int = 10):
        self.clipcap.maybe_patch_gpt(batch_size * prefix_len)

    def unpatch_model(self):
        self.clipcap.maybe_unpatch_gpt()
