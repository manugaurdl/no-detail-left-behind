# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import egg.core as core
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.dataloaders import (
    CocoWrapper,
    FlickrWrapper,
    NoCapsWrapper,
    get_transform,
)
from egg.zoo.emergent_captioner.finetuning.blip import BlipSender
from egg.zoo.emergent_captioner.finetuning.game import ReinforceCaptionGame
from egg.zoo.emergent_captioner.finetuning.losses import Loss
from egg.zoo.emergent_captioner.finetuning.opts import get_common_opts
from egg.zoo.emergent_captioner.finetuning.receiver import ClipReceiver
from egg.zoo.emergent_captioner.finetuning.clipcap import ClipCapSender
from egg.zoo.emergent_captioner.utils import get_sha, log_stats

DATASET2NEG_PATHS = {
    "flickr": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/test_flickr.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/test_flickr.nns.pt",
    ),
    "coco": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/test_coco.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/test_coco.nns.pt",
    ),
    "nocaps": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/all.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/all.nns.pt",
    ),
}


class FullSetLoss(Loss):
    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        elem_idxs = img_idxs.squeeze()

        sims = text_feats @ self.emb.to(text_feats.device).t()
        loss = F.cross_entropy(sims, elem_idxs, reduction="none")
        acc = (sims.argmax(dim=1) == elem_idxs).detach().float()

        return loss, {"acc": acc}


class HumanCaptionSender(nn.Module):
    def forward(self, x, aux_input=None):
        if isinstance(aux_input["captions"], list):
            return aux_input["captions"][0]

        return aux_input["captions"]


class HumanGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        logging_strategy: LoggingStrategy = None,
    ):
        super(HumanGame, self).__init__()

        self.train_logging_strategy = LoggingStrategy.minimal()
        self.test_logging_strategy = (
            LoggingStrategy.minimal() if logging_strategy is None else logging_strategy
        )

        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        captions = self.sender(sender_input, aux_input)

        with torch.no_grad():
            text_feats, img_feats = self.receiver(captions, receiver_input, aux_input)
            loss, aux_info = self.loss(text_feats, img_feats, labels, aux_input)

        loss, aux_info = self.loss(text_feats, img_feats, labels, aux_input)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=captions,
            receiver_output=None,
            message_length=None,
            aux=aux_info,
        )
        return loss.mean(), interaction


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    if opts.load_from_checkpoint:
        checkpoint = torch.load(opts.load_from_checkpoint)

        try:  # old models don't have opts stored in the checkpoints
            loaded_opts = checkpoint.opts
        except AttributeError:
            pass
        else:
            loaded_opts.checkpoint_dir = opts.checkpoint_dir
            loaded_opts.eval_datasets = opts.eval_datasets
            opts = loaded_opts

        try:
            del checkpoint.model_state_dict["loss.logit_scale"]
        except KeyError:
            pass

    human_sender = HumanCaptionSender()

    if opts.captioner_model == "clipcap":
        model_sender = ClipCapSender(
            clip_model=opts.sender_clip_model,
            clipcap_path=opts.clipcap_model_path,
            do_sample=opts.do_sample,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
        )
    elif opts.captioner_model.lower() == "blip":
        model_sender = BlipSender(
            blip_model=opts.blip_model,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
            freeze_visual_encoder=opts.freeze_blip_visual_encoder,
        )
    else:
        raise RuntimeError

    receiver = ClipReceiver(clip_model=opts.recv_clip_model)

    data_kwargs = dict(
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        seed=opts.random_seed,
    )

    for dataset in opts.eval_datasets:
        emb, nns = DATASET2NEG_PATHS.get(dataset, (None, None))

        assert emb and nns
        loss = FullSetLoss(emb, nns)

        # remember that with non-diff losses you should use a wrapper around recv
        model_game = ReinforceCaptionGame(
            sender=model_sender,
            receiver=receiver,
            loss=loss,
            baseline=opts.baseline,
            kl_div_coeff=opts.kl_div_coeff,
        )

        # human_game = HumanGame(sender=human_sender, receiver=receiver, loss=loss)
        _ = HumanGame(sender=human_sender, receiver=receiver, loss=loss)

        if opts.captioner_model == "clipcap":
            model_game.sender.patch_model()

        trainer = core.Trainer(
            game=model_game,
            optimizer=torch.optim.Adam(model_game.sender.parameters(), lr=opts.lr),
            train_data=None,
            debug=opts.debug,
        )

        if opts.captioner_model == "clipcap":
            trainer.game.sender.patch_model()

        assert dataset in ["coco", "flickr", "nocaps"]
        wrappers = {
            "coco": CocoWrapper,
            "flickr": FlickrWrapper,
            "nocaps": NoCapsWrapper,
        }

        wrapper = wrappers[dataset.lower()]()
        split = "all" if dataset == "nocaps" else "test"
        test_loader = wrapper.get_split(split=split, **data_kwargs)
        _, interaction = trainer.eval(test_loader)
        log_stats(interaction, f"Model performance: {dataset.upper()} TEST SET")

        """
        trainer = core.Trainer(
            game=human_game,
            optimizer=torch.optim.Adam(human_game.sender.parameters(), lr=opts.lr),
            train_data=None,
            debug=opts.debug,
        )

        # _, human_interaction = trainer.eval(test_loader)
        # log_stats(human_interaction, f"Human performance: {dataset.upper()} TEST SET")
        """

    end = time.time()

    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
