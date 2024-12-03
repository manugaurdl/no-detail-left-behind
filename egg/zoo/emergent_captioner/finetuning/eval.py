# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import torch

import egg.core as core
from egg.zoo.emergent_captioner.dataloaders import (
    CocoWrapper,
    ConcadiaWrapper,
    ConceptualCaptionsWrapper,
    FlickrWrapper,
    ImageCodeWrapper,
    NoCapsWrapper,
    VizWizWrapper,
    get_transform,
)
from egg.zoo.emergent_captioner.finetuning.game import build_game
from egg.zoo.emergent_captioner.finetuning.opts import get_common_opts
from egg.zoo.emergent_captioner.evaluation.evaluate_nlg import compute_nlg_metrics
from egg.zoo.emergent_captioner.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    setup_for_distributed,
    store_job_and_task_id,
)


def prepare_for_nlg_metrics(predictions, gt, multi_reference):
    preds = {}
    gold = {}
    for i, (p, gg) in enumerate(zip(predictions, gt)):
        preds[i] = [{"caption": p}]
        if multi_reference:
            gold[i] = [{"caption": g} for g in gg]
        else:
            gold[i] = [{"caption": gg}]
    return preds, gold


def extract_captions(interaction):
    captions = [
        caption.strip().replace("\n", "")
        for batch_captions in interaction.message
        for caption in batch_captions
    ]
    return captions


def extract_gt(interaction, multi_reference: bool = False):
    all_captions = []
    for batch in interaction.aux_input["captions"]:
        if multi_reference:
            for batch_captions in zip(*batch):
                all_captions.append(list(batch_captions))
        else:
            all_captions.extend(batch)

    return all_captions


def main(params):
    start = time.time()
    # params.extend(["--eval_datasets", ["coco"]])
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    setup_for_distributed(opts.distributed_context.is_leader)
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

    game = build_game(opts)
    if opts.captioner_model == "clipcap":
        game.sender.patch_model()

    trainer = core.Trainer(
        game=game,
        optimizer=torch.optim.Adam(game.sender.parameters(), lr=opts.lr),
        train_data=None,
        debug=opts.debug,
    )
    if opts.captioner_model == "clipcap":
        trainer.game.sender.patch_model()

    data_kwargs = dict(
        batch_size=opts.batch_size,
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        num_workers=opts.num_workers,
        seed=opts.random_seed,
    )

    logging.disable(level=logging.INFO)
    for dataset in opts.eval_datasets:
        if dataset in [
            "concadia",
            "conceptual",
            "coco",
            "flickr",
            "imagecode",
            "vizwiz",
        ]:
            wrappers = {
                "coco": CocoWrapper,
                "concadia": ConcadiaWrapper,
                "conceptual": ConceptualCaptionsWrapper,
                "flickr": FlickrWrapper,
                "imagecode": ImageCodeWrapper,
                "vizwiz": VizWizWrapper,
            }

            dataset = "coco"
            wrapper = wrappers[dataset.lower()]()
            test_loader = wrapper.get_split(split="test", **data_kwargs)
            _, interaction = trainer.eval(test_loader)
            log_stats(interaction, f"{dataset.upper()} TEST SET")
            dump_interaction(interaction, opts, name=f"{dataset.lower()}_")

            multi_reference = dataset not in ["conceptual", "concadia", "imagecode"]
            gt = extract_gt(interaction, multi_reference=multi_reference)
            preds = extract_captions(interaction)
            preds, gt = prepare_for_nlg_metrics(preds, gt, multi_reference)

            print(f"EVALUATING {dataset}")
            compute_nlg_metrics(preds, gt)

        elif dataset == "nocaps":
            nocaps_wrapper = NoCapsWrapper()
            for split in ["all", "in-domain", "near-domain", "out-domain"]:
                test_loader = nocaps_wrapper.get_split(split=split, **data_kwargs)
                _, interaction = trainer.eval(test_loader)
                log_stats(interaction, f"NOCAPS {split} TEST SET")
                dump_interaction(interaction, opts, name=f"nocaps_{split}_")

                gt = extract_gt(interaction, multi_reference=True)
                preds = extract_captions(interaction)
                preds, gt = prepare_for_nlg_metrics(preds, gt, multi_reference=True)

                print(f"EVALUATING nocaps {split}")
                compute_nlg_metrics(preds, gt)
        else:
            print(f"Cannot recognize {dataset} dataset. Skipping...")

    end = time.time()

    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
