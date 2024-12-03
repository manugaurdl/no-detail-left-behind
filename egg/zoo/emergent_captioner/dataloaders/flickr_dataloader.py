# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
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


class FlickrDataset:
    def __init__(self, root, samples, transform):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, captions, image_id = self.samples[idx]

        image = Image.open(os.path.join(self.root, file_path)).convert("RGB")
        sender_input, recv_input = self.transform(image)

        aux = {"img_id": image_id, "captions": captions}

        return sender_input, torch.tensor([idx]), recv_input, aux


class FlickrWrapper:
    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/flickr30k"
        self.dataset_dir = Path(dataset_dir)

        self.split2samples = self._load_splits()

    def _load_splits(self):
        with open(self.dataset_dir / "dataset_flickr30k.json") as f:
            annotations = json.load(f)
        split2samples = defaultdict(list)
        for img_ann in annotations["images"]:
            file_path = self.dataset_dir / "Images" / img_ann["filename"]
            captions = [x["raw"] for x in img_ann["sentences"]]
            img_id = img_ann["imgid"]
            split = img_ann["split"]

            split2samples[split].append((file_path, captions, img_id))
        if "restval" in split2samples:
            split2samples["train"] += split2samples["restval"]

        for k, v in split2samples.items():
            print(f"| Split {k} has {len(v)} elements.")
        return split2samples

    def get_split(
        self,
        split: str,
        batch_size: int,
        transform: Callable,
        num_workers: int = 8,
        shuffle: bool = None,
        seed: int = 111,
    ):

        samples = self.split2samples[split]
        assert samples, f"Wrong split {split}"

        ds = FlickrDataset(self.dataset_dir, samples, transform=transform)

        sampler = None
        if dist.is_initialized():
            if shuffle is None:
                shuffle = split != "test"
            sampler = MyDistributedSampler(
                ds, shuffle=shuffle, drop_last=True, seed=seed
            )

        if shuffle is None:
            shuffle = split != "test" and sampler is None

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return loader
