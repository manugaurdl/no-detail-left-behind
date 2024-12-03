# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import uuid
from pathlib import Path

import torch

DATASET2NEG_PATHS = {
    "flickr": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/train_flickr.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/train_flickr.nns.pt",
    ),
    "coco": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/train_coco.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/train_coco.nns.pt",
    ),
    "conceptual": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/train_conceptual.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/train_conceptual.nns.pt",
    ),
}


def print_grad_info(model):
    grad = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"GRAD {grad}")


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def store_job_and_task_id(opts):
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None or task_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
        task_id = os.environ.get("SLURM_PROCID", 0)

    opts.job_id = job_id
    opts.task_id = task_id
    return job_id, task_id


def log_stats(interaction, mode):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
        dump.update(dict(mode=mode))
        print(json.dumps(dump), flush=True)


def dump_interaction(interaction, opts, name=""):
    if opts.checkpoint_dir and opts.distributed_context.is_leader:
        output_path = Path(opts.checkpoint_dir) / "interactions"
        output_path.mkdir(exist_ok=True, parents=True)
        interaction_name = f"interaction_{name}{opts.job_id}_{opts.task_id}"

        interaction.aux_input["args"] = opts
        torch.save(interaction, output_path / interaction_name)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def setup_for_distributed(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
