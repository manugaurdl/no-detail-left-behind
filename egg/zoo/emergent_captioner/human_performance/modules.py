# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import clip
import torch
import torch.nn as nn

from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class ClipReceiver(torch.nn.Module):
    def __init__(self, clip_model: str):
        super(ClipReceiver, self).__init__()
        self.clip = clip.load(clip_model)[0]
        convert_models_to_fp32(self.clip)
        self.clip.eval()

    def forward(self, message, images, aux_input=None):
        text = clip.tokenize(message, truncate=True).to(images.device)
        with torch.no_grad():
            _, clip_logits = self.clip(images, text)
        return clip_logits


class HumanCaptionSender(nn.Module):
    def forward(self, x, aux_input=None):
        if isinstance(aux_input["captions"], list):
            return aux_input["captions"][0]

        return aux_input["captions"]


class ZeroShotCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        logging_strategy: LoggingStrategy = None,
    ):
        super(ZeroShotCaptionGame, self).__init__()

        self.train_logging_strategy = LoggingStrategy.minimal()
        self.test_logging_strategy = (
            LoggingStrategy.minimal() if logging_strategy is None else logging_strategy
        )

        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        message_length = torch.Tensor([len(x) for x in message]).int()

        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=message,
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )
        return loss.mean(), interaction


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):
    batch_size = receiver_output.shape[0]
    labels = torch.arange(batch_size, device=receiver_output.device)

    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return torch.zeros(1), {"acc": acc}
