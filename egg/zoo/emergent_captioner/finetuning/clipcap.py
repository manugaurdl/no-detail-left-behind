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
# from LLaVA_pp.LLaVA.llava.model import *

### LLaVa
# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
#     IMAGE_PLACEHOLDER,
# )
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import (
#     process_images,
#     tokenizer_image_token,
#     get_model_name_from_path,
# )

# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from transformers import (AutoTokenizer, BitsAndBytesConfig, StoppingCriteria,
#                           StoppingCriteriaList, TextStreamer)

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

    """
    output learnable tokens from transformer. They will learn to exctract visual information required for captioning. Kinda like cross-attention
    """
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
                # at test time we use beam search regardless of the decoding method
                # used at training time
                print("BEAM SEARCH GENERATION")
                # print(f"USING GPU:{os.environ['LOCAL_RANK']}")

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

            # logits after generation bruh
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
            try:
                log_probs = torch.gather(logits, dim=2, index=indices.unsqueeze(2)).squeeze(-1) # (B, 10) : log prob for each sampled policy word
                log_probs *= mask # everything after "." is zeroed
                log_probs = log_probs.sum(1) / msg_lengths #averaged
            except:
                breakpoint()
            # put captions in textual form
            decoded_captions = self.tokenizer.batch_decode(
                indices,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # compute kl loss
            # print(f"-------------->{indices.device}")
            # print(f"-------------->{logits.device}")

            # kl_div = self.kl_regularizer.compute_kl_loss(indices, logits)
            # kl_div *= mask
            # kl_div = kl_div.sum(-1) / msg_lengths
            
            self.do_sample = False

        return decoded_captions, log_probs, torch.randn(1)

    def maybe_patch_gpt(self, max_embeddings):
        if not getattr(self.gpt, "_patched", False):
            self.gpt._patched = True
            #increase gpt nn.embedding from vocab size --> + 5000
            self.gpt.resize_token_embeddings(len(self.tokenizer) + max_embeddings)
            
            # bias for output embeddings (768) added (55257)
            # its requires grad is False?
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
                desired_format_state_dict = torch.load(official_clipcap_weights)
                saved_state_dict = torch.load(clipcap_path)
                
                
                ################################################################################
                
                #### LOADING PREV CODEBASE MLE MODEL
                # state_dict = torch.load(clipcap_path)['model_state_dict']
                # state_dict["clip_project.model.0.weight"] = state_dict["mapping_network.model.0.weight"]
                # state_dict["clip_project.model.2.weight"] = state_dict["mapping_network.model.2.weight"]
                # state_dict["clip_project.model.0.bias"] = state_dict["mapping_network.model.0.bias"]
                # state_dict["clip_project.model.2.bias"] = state_dict["mapping_network.model.2.bias"]
                # del state_dict["mapping_network.model.0.weight"]
                # del state_dict["mapping_network.model.2.weight"]
                # del state_dict["mapping_network.model.0.bias"]
                # del state_dict["mapping_network.model.2.bias"]
                
                #### LOAD EGG MLE MODEL
                state_dict = {}
                for idx, k in enumerate(desired_format_state_dict.keys()):
                    state_dict[k] = saved_state_dict["sender.clipcap." + k]

                ##################################################################################
                self.clipcap.load_state_dict(state_dict)
                # self.clipcap.load_state_dict(desired_format_state_dict)


    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None, CIDER_OPTIM= False, use_fp16 = False, greedy_baseline = False, train_method = None):
        #if loading CLIP on GPU
        image_feats = self.clip.visual(images)

        #if lazy loading clip_feats
        # image_feats = images.squeeze(1)
        # if image_feats.dtype != torch.float16:
        #     print("img feat not fp16")
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
################################################################################################################################################################

class LLavaSender(nn.Module):
    def __init__(
        self,
        # clip_model: str,
        # official_clipcap_weights : str,
        train_method : str,
        config : dict,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(LLavaSender, self).__init__()

        assert max_len < 75  # clip maximum context size

        #Llava args -----------------
        model_path = "liuhaotian/llava-v1.6-mistral-7b"
        model_base = None
        conv_mode = None
        sep = ","
        self.temperature = 0.2
        self.top_p=None
        self.num_beams=1
        self.max_new_tokens= 64
        qs = 'Describe the image briefly.'
        #--------------------
        disable_torch_init()

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name
        )
        
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs # <--

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if conv_mode is not None and conv_mode != conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, conv_mode, conv_mode
                )
            )
        else:
            conv_mode = conv_mode # <--

        ## Conversation templates
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # #####delete this bs
        # from PIL import Image

        # import requests
        # from PIL import Image
        # from io import BytesIO
        # import re


        # def image_parser(args,sep):
        #     out = image_file.split(sep)
        #     return out


        # def load_image(image_file):
        #     if image_file.startswith("http") or image_file.startswith("https"):
        #         response = requests.get(image_file)
        #         image = Image.open(BytesIO(response.content)).convert("RGB")
        #     else:
        #         image = Image.open(image_file).convert("RGB")
        #     return image


        # def load_images(image_files):
        #     out = []
        #     for image_file in image_files:
        #         image = load_image(image_file)
        #         out.append(image)
        #     return out

        # ###### delete this bs
        # # image_files = image_parser(args)
        # image_files = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
        # images = load_images(image_files)


        self.input_ids = (
        tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
        )



#---------------------------------------------------------------------------------------------------------------------------


    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None, CIDER_OPTIM= False, greedy_baseline = False, train_method = None):

        s = time.time()
        # with torch.inference_mode():
        gen_dict = self.model.generate(
            torch.cat([self.input_ids]*images.shape[0]),
            # images=images_tensor,
            images = images,
            image_sizes=[(640, 480)]*images.shape[0],
            do_sample=False if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            output_scores=True, 
            return_dict_in_generate=True,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        print(f"time elapsed generating : {time.time() - s}")
        gen_tokens = gen_dict.sequences
        scores = gen_dict.scores # {max_len of gen_tokens} tensors : each B X vocab_size
                            #get logprobs for each token

        B, max_len_batch = gen_tokens.shape
        end_of_caption = gen_tokens == self.tokenizer.eos_token_id #50256 padding tokens

        # compute_mask and msg_lengths
        max_k = max_len_batch #len of longest generation in batch i.e 10
        extra_tokens = end_of_caption.cumsum(dim=1) > 0
        msg_lengths = max_k - (extra_tokens).sum(dim=1)
        # msg_lengths.add_(1).clamp_(max=max_k) #lengths increased by 1?

        mask = (extra_tokens == 0).float()
        
        scores = torch.stack(scores)[:,:, : len(self.tokenizer)].permute(1,0,2) #(B*max_batch_len)   # i1 i2 i1 i2 i1 i2 ..... 12 times
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        logprobs = torch.gather(logprobs, dim =-1, index = gen_tokens[:,:-1].unsqueeze(-1)).squeeze(-1)
        logprobs *= mask[:, :-1]
        logprobs = logprobs.sum(1) / msg_lengths


        decoded_captions = self.tokenizer.batch_decode(
            gen_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_captions = [_.strip() for _ in decoded_captions]

        return decoded_captions, logprobs, torch.randn(1)

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

    # def train(self, mode: bool = True):
    #     self.training = mode
    #     self.clipcap.train(mode)
    #     return self


    ####################################################################################################################################################################################


class LLavaPhi(nn.Module):


    def find_all_linear_names(self, model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["vision_tower"]# "multi_modal_projector", "language_model"
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                # names = name.split('.')
                # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                lora_module_names.add(name)

        # if lm_head:
        #     continue
        # else:
        #     if 'lm_head' in lora_module_names: # needed for 16-bit
        #         lora_module_names.remove('lm_head')
        # for name, module in model.named_modules():
        #     if "lm_head" in name:
        #         lora_module_names.add(name)
        return list(lora_module_names)

    def __init__(
        self,
        train_method : str,
        config : dict,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(LLavaPhi, self).__init__()

        assert max_len < 75  # clip maximum context size

        self.recv_clip_model = config['opts']['recv_clip_model']
        #Llava-phi args -----------------
        model_id = "xtuner/llava-phi-3-mini-hf"
        self.temperature = config['temp']
        self.top_p=None
        self.num_beams=1
        self.max_new_tokens= config['max_new_tokens']
        self.lora_rank= config["lora_rank"]
        self.n_layers = config['n_layers']
        self.lm_head = config['lm_head']
        #--------------------
        # disable_torch_init()
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
            ).to("cuda")

        # self.model = LlavaPhiForCausalLM.from_pretrained(
        #         "microsoft/Phi-3-mini-4k-instruct",
        #         attn_implementation="flash_attention_2",
        #         torch_dtype=torch.bfloat16, 
        #         low_cpu_mem_usage=True,
        #     )



        self.processor = AutoProcessor.from_pretrained(model_id)
        self.prompt = "<|user|>\n<image>\nDescribe the image briefly.<|end|>\n<|assistant|>\n"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

#       #PEFT lora
        
        lin_layers = self.find_all_linear_names(self.model) #got all LLM layers
        
        last_llm_layers = []
        for l in lin_layers:
            if self.n_layers == 9 and not self.lm_head:
                if ("language_model.model.layers.23" in l) or ("language_model.model.layers.24" in l) or ("language_model.model.layers.25" in l) or ("language_model.model.layers.26" in l) or ("language_model.model.layers.27" in l) or ("language_model.model.layers.28" in l) or ("language_model.model.layers.29" in l) or ("language_model.model.layers.30" in l) or ("language_model.model.layers.31" in l) or ("lm_head" in l):
                    last_llm_layers.append(l)
            elif self.lm_head:
                if ("language_model.model.layers.27" in l) or ("language_model.model.layers.28" in l) or ("language_model.model.layers.29" in l) or ("language_model.model.layers.30" in l) or ("language_model.model.layers.31" in l) or ("lm_head" in l):
                    last_llm_layers.append(l)
            else:
                if ("language_model.model.layers.27" in l) or ("language_model.model.layers.28" in l) or ("language_model.model.layers.29" in l) or ("language_model.model.layers.30" in l) or ("language_model.model.layers.31" in l):
                    last_llm_layers.append(l)
        
        lora_config = LoraConfig(
        r=self.lora_rank,
        lora_alpha=16,
        # lora_dropout=0.1,
        target_modules= last_llm_layers,
        bias="none",
        # task_type="CAUSAL_LM",
    )
        self.model = get_peft_model(self.model, lora_config)
        # self.model = self.model.merge_and_unload()


#---------------------------------------------------------------------------------------------------------------------------


    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None, CIDER_OPTIM= False, greedy_baseline = False, train_method = None, use_fp16 = False):

        context = autocast(enabled=True, dtype=torch.bfloat16) if use_fp16 else nullcontext()

        with context:


            images = [Image.open(cocoid2img(cocoid)).convert("RGB") for cocoid in aux_input['cocoid']]
            inputs = self.processor([self.prompt]*len(images), images=images, return_tensors="pt", padding=True).to("cuda", dtype=torch.float16)
            # s = time.time()
            # with torch.inference_mode():
            #     gen_dict = self.model.generate(
            #         **inputs,
            #         do_sample=True if self.temperature > 0 else False,
            #         # do_sample=False,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #         output_scores=True, 
            #         return_dict_in_generate=True,
            #         # num_beams=self.num_beams,
            #         max_new_tokens=self.max_new_tokens,
            #         use_cache=True,
            #     )

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=True if self.temperature > 0 else False,
                    # do_sample=False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    # num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                )


    ##########################################################################################################################################################
            # GENERATING WITH GRADIENTS

            # gen_tokens = gen_dict.sequences[: , -self.max_new_tokens:]
            # scores = gen_dict.scores #num tokens generated = max_new_tokens = len(scores)

            # B, max_len_batch = gen_tokens.shape
            # end_of_caption = gen_tokens == self.tokenizer.eos_token_id #50256 padding tokens

            # # compute_mask and msg_lengths
            # max_k = max_len_batch #len of longest generation in batch i.e 10
            # extra_tokens = end_of_caption.cumsum(dim=1) > 0
            # msg_lengths = max_k - (extra_tokens).sum(dim=1)
            # # msg_lengths.add_(1).clamp_(max=max_k) #lengths increased by 1?

            # mask = (extra_tokens == 0).float()
            
            # scores = torch.stack(scores)[:,:, : len(self.tokenizer)].permute(1,0,2) #(B*max_batch_len)   # i1 i2 i1 i2 i1 i2 ..... 12 times
            # logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
            # logprobs = torch.gather(logprobs, dim =-1, index = gen_tokens.unsqueeze(-1)).squeeze(-1)
            # logprobs *= mask
            # logprobs = logprobs.sum(1) / msg_lengths

##########################################################################################################################################################
        # TRYING WITH NO GRADIENTS
        """
        input_ids = prompt tokens 
        
        class 'transformers.models.llava.modeling_llava.LlavaForConditionalGeneration

        """
        
        context = autocast(enabled=True, dtype=torch.bfloat16) if use_fp16 else nullcontext()
        with context:
            inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])#.half() # B, 11, 3072 ("Describe the image" has 11 tokens)
            
            image_outputs = self.model.vision_tower(inputs['pixel_values'], output_hidden_states=True) # ['last_hidden_state', 'pooler_output', 'hidden_states']
            selected_image_feature = image_outputs.hidden_states[-2]   # select -2 layer from 25 layers
            selected_image_feature = selected_image_feature[:, 1:] # B, N, D_clip --> skip first token (B, N-1, D_clip)
            image_features = self.model.multi_modal_projector(selected_image_feature) # B, N, D_lm
            labels = None
            attention_mask = torch.ones(inputs['input_ids'].shape)
            #image_feats : (B, 576,D)
            #prompt feats : (B,11,D)
            #cat feats : (B, 586, D)
            inputs_embeds, attention_mask, labels, position_ids = self.model._merge_input_ids_with_image_features(
                image_features, inputs_embeds, inputs['input_ids'], attention_mask, labels
            )

            prefix_len = inputs_embeds.shape[1] # text + img promp

            # get embeds of generated tokens 
            prompt_len = inputs['input_ids'].shape[-1]
            indices = generated[:, prompt_len:].clone() # generated (B, max_batch_len) tokens. After "."/13 padded with "<eos>/50256
            suffix = self.model.get_input_embeddings()(indices) # B, 10, 768 embd. 
            
            #get prompt + gen embeds
            inputs_embeds = torch.cat([inputs_embeds, suffix], dim=1) # B, 20,768 emb

            logits = self.model.language_model(inputs_embeds=inputs_embeds) ### self.language_model
            logits = logits[0][:, prefix_len - 1 : -1, : len(self.tokenizer)] # extract last logit from --> logits[0] : (B, max_batch_len, 55257) 
            logits = logits/(self.temperature if self.temperature > 0 else 1.0)
            logits = logits.log_softmax(-1)

            # compute_mask and msg_lengths
            max_k = indices.size(1) #len of longest generation in batch i.e 10
            end_of_caption = indices == self.tokenizer.eos_token_id #50256 padding tokens
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
            decoded_captions = [_.strip() for _ in decoded_captions]
            # print(decoded_captions[10:20])
            # print(f"time elapsed generating : {time.time() - s}")
        
        if self.recv_clip_model=="ViT-L/14@336px":
            receiver_input = inputs['pixel_values']
        else:
            receiver_input = images
        
        return decoded_captions, log_probs, torch.randn(1), receiver_input

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


