# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn

import clip

from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class ClipSender(nn.Module):
    def __init__(self, clip_model):
        super(ClipSender, self).__init__()
        # self.clip, self.clip_preproc = clip.load(clip_model)
        self.clip = clip_model
        convert_models_to_fp32(self.clip)
        # self.clip.eval()

    def forward(self, message, images, aux_input=None, img_feats = True):

        text = clip.tokenize(message, truncate=True).to(images.device)
                
        text_features = self.clip.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # text_features = text_features * self.clip.logit_scale.exp()
        
        if not img_feats:
            return text_features
        image_features = self.clip.encode_image(images)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # image_features might be used for computing hard distractors
        # hence scaling text features now
        # this is the equivalent to computing cosine/dot product similarity and then scaling
        return text_features, image_features

        # text = clip.tokenize(message, truncate=True).to(images.device)
        # _, clip_logits = self.clip(images, text)
        # return clip_logits

        # with torch.no_grad():
        #     text_feats = self.encode_captions(text)
        # return text_feats
