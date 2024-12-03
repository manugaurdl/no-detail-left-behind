# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist


from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler


class VizWizDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        transform,
    ):
        assert split in ["train", "test"]

        self.dataset_dir = os.path.realpath(dataset_dir)
        with open(dataset_dir / "annotations" / f"{split}.json", "r") as fd:
            data = json.load(fd)

        img_info = defaultdict(list)
        for img_dict in data["images"]:
            img_info[img_dict["id"]].append(img_dict["file_name"])

        ann_info = defaultdict(list)
        for ann_dict in data["annotations"]:
            caption, img_id = ann_dict["caption"], ann_dict["image_id"]
            ann_info[img_id].append(caption)

        assert len(img_info) == len(ann_info)
        for img_id, captions in ann_info.items():
            img_info[img_id].extend(captions)

        self.samples = [(k, v[0], v[1:]) for k, v in img_info.items()]

        self.split = "val" if split == "test" else "train"

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, file_name, captions = self.samples[idx]

        file_path = os.path.join(self.dataset_dir, "images", self.split, file_name)
        image = Image.open(file_path).convert("RGB")

        sender_input, recv_input = self.transform(image)

        aux = {"img_id": torch.tensor([img_id]), "captions": captions}

        return sender_input, torch.tensor([idx]), recv_input, aux


class VizWizWrapper:
    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/vizwiz"
        self.dataset_dir = Path(dataset_dir)

    def get_split(
        self,
        split: str,
        transform: Callable,
        batch_size: int = 4,
        num_workers: int = 8,
        shuffle: bool = None,
        seed: int = 111,
    ):
        ds = VizWizDataset(self.dataset_dir, split=split, transform=transform)

        sampler = None
        if dist.is_initialized():
            if shuffle is None:
                shuffle = split != "test"
            sampler = MyDistributedSampler(
                ds, shuffle=shuffle, drop_last=True, seed=seed
            )

        print(f"| Split {split} has {len(ds)} elements.")

        if shuffle is None:
            shuffle = split != "test" and sampler is None

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=sampler is None and split != "test",
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        return loader


if __name__ == "__main__":
    from utils import get_transform

    w = VizWizWrapper()
    dl = w.get_split(split="test", num_workers=0, transform=get_transform(224))
    for i, elem in enumerate(dl):
        breakpoint()
        if i == 10:
            break
        continue
