# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time

import torch

import egg.core as core
from egg.zoo.emergent_captioner.dataloaders import (
    ConcadiaWrapper,
    ConceptualCaptionsWrapper,
    CocoWrapper,
    FlickrWrapper,
    ImageCodeWrapper,
    NoCapsWrapper,
    VizWizWrapper,
    get_transform,
)
from egg.zoo.emergent_captioner.human_performance.modules import (
    ClipReceiver,
    HumanCaptionSender,
    ZeroShotCaptionGame,
    loss,
)
from egg.zoo.emergent_captioner.utils import get_sha, log_stats, store_job_and_task_id


def get_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled and on only 10 batches",
    )
    parser.add_argument(
        "--recv_clip_model",
        choices=["ViT-B/16", "ViT-B/32"],
        default="ViT-B/32",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=8)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def main(params):
    start = time.time()

    opts = get_opts(params)
    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    sender = HumanCaptionSender()
    receiver = ClipReceiver(clip_model=opts.recv_clip_model)
    game = ZeroShotCaptionGame(sender, receiver, loss)

    print(f"| Evaluating with CLIP {opts.recv_clip_model}")
    trainer = core.Trainer(
        game=game,
        optimizer=torch.optim.Adam(game.parameters(), lr=opts.lr),
        train_data=None,
        debug=opts.debug,
    )

    wrappers = {
        "coco": CocoWrapper,
        "concadia": ConcadiaWrapper,
        "conceptual": ConceptualCaptionsWrapper,
        "flickr": FlickrWrapper,
        "imagecode": ImageCodeWrapper,
        "vizwiz": VizWizWrapper,
    }

    data_kwargs = dict(
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        transform=get_transform(opts.image_size),
        seed=opts.random_seed,
    )
    for dataset, wrapper in wrappers.items():
        print(f"| Evaluating dataset {dataset}")
        w = wrapper()
        test_loader = w.get_split(split="test", **data_kwargs)
        _, interaction = trainer.eval(test_loader)
        log_stats(interaction, f"{dataset.upper()} TEST SET")

    nocaps_wrapper = NoCapsWrapper()
    for split in ["in-domain", "near-domain", "out-domain", "all"]:
        test_loader = nocaps_wrapper.get_split(split=split, **data_kwargs)
        _, interaction = trainer.eval(test_loader)
        log_stats(interaction, f"NOCAPS {split} TEST SET")

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
