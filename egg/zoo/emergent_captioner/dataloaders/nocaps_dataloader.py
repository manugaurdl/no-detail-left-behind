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


class NoCapsDataset:
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

        aux = {
            "img_id": torch.tensor([image_id]),
            "captions": captions,
            "filepath": str(file_path),
        }

        return sender_input, torch.tensor([idx]), recv_input, aux


class NoCapsWrapper:
    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = "/checkpoint/rdessi/datasets/open_images"
        self.dataset_dir = Path(dataset_dir)

        self.split2samples = self._load_splits()

    def _load_splits(self):
        with open(self.dataset_dir / "nocaps_val_4500_captions.json") as f:
            metadata = json.load(f)
        split2samples = defaultdict(list)

        img_data, img_ann = metadata["images"], metadata["annotations"]
        for i, sample_info in enumerate(img_data):
            start, end = i * 10, i * 10 + 10
            caption_info = img_ann[start:end]
            assert all([sample_info["id"] == x["image_id"] for x in caption_info])

            file_path = self.dataset_dir / "validation" / sample_info["file_name"]
            img_id = sample_info["id"]
            split = sample_info["domain"]
            captions = [x["caption"] for x in caption_info]

            split2samples[split].append((file_path, captions, img_id))

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

        assert split in ["in-domain", "near-domain", "out-domain", "all"]

        if split == "all":
            datasets = []
            for s in ["in-domain", "near-domain", "out-domain"]:
                samples = self.split2samples[s]
                d = NoCapsDataset(self.dataset_dir, samples, transform=transform)
                datasets.append(d)

            ds = torch.utils.data.ConcatDataset(datasets)
        else:
            samples = self.split2samples[split]
            ds = NoCapsDataset(self.dataset_dir, samples, transform=transform)

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader


if __name__ == "__main__":
    wrapper = NoCapsWrapper()
    dl = wrapper.get_split(
        split="near-domain",
        batch_size=10,
        image_size=224,
        shuffle=False,
        num_workers=0,
    )

    for i, elem in enumerate(dl):
        breakpoint()
