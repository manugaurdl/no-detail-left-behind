# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
import torch.nn as nn

# from lavis.models import load_model


class BlipSender(nn.Module):
    def __init__(
        self,
        blip_model: str = "base_coco",
        beam_size: int = 5,
        max_len: int = 20,
        freeze_visual_encoder: bool = False,
    ):
        super(BlipSender, self).__init__()
        model = load_model(name="blip_caption", model_type=blip_model)

        self.tokenizer = model.tokenizer
        self.prompt = self.tokenizer(model.prompt).input_ids[:-1]
        self.prompt[0] = self.tokenizer.bos_token_id

        self.model_txt = model.text_decoder
        self.model_txt.config.eos_token_id = self.tokenizer.sep_token_id
        self.model_txt.config.pad_token_id = self.tokenizer.pad_token_id

        assert beam_size == 1
        self.beam_size = beam_size

        self.max_len = max_len

        self.model_img = model.visual_encoder
        self.model_img.eval()

        self.freeze_visual_encoder = freeze_visual_encoder

        if self.freeze_visual_encoder:
            for p in self.model_img.parameters():
                p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def prefix_len(self):
        return len(self.prompt)

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.model_txt.named_parameters()
        if self.freeze_visual_encoder:
            return self.model_txt.named_parameters()
        return super().named_parameters(prefix=prefix, recurse=recurse)

    def parameters(self, recurse: bool = True):
        return self.model_txt.parameters()
        if self.freeze_visual_encoder:
            return self.model_txt.parameters()
        return super().parameters(recurse=recurse)

    def train(self, mode: bool = True):
        if self.freeze_visual_encoder:
            self.training = mode
            self.model_img.eval()
            self.model_txt.train(mode)
            return self
        return super().train(mode=mode)

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):
        with torch.no_grad():
            feats_img = self.model_img(images)
            attns_img = torch.ones(feats_img.size()[:-1], dtype=torch.long).to(
                self.device
            )

            model_kwargs = {
                "encoder_hidden_states": feats_img,
                "encoder_attention_mask": attns_img,
            }

            prompts = (
                torch.tensor(self.prompt)
                .unsqueeze(0)
                .repeat(images.size(0), 1)
                .to(self.device)
            )

            captions = self.model_txt.generate(
                input_ids=prompts,
                num_beams=self.beam_size,
                do_sample=False,
                early_stopping=False,
                max_length=self.max_len,
                **model_kwargs
            )

            mask = captions != self.model_txt.config.pad_token_id
            mask = mask[:, self.prefix_len :]
            msg_lengths = mask.long().sum(-1)

        logits = self.model_txt(captions, **model_kwargs).logits
        log_probs = logits.log_softmax(-1)[:, self.prefix_len - 1 : -1]
        log_probs = log_probs.gather(
            2, captions[:, self.prefix_len :].unsqueeze(2)
        ).squeeze(-1)
        log_probs *= mask.to(log_probs.dtype)
        log_probs = log_probs.sum(-1) / msg_lengths.to(log_probs.dtype)

        captions = captions[:, self.prefix_len :]
        decoded_captions = self.tokenizer.batch_decode(
            captions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return decoded_captions, log_probs, torch.zeros(1).to(self.device)
