# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Optional
from PIL import Image

import torch
import torch.distributed as dist
from torchvision.datasets import VisionDataset

from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler


class ConceptualCaptionsDataset(VisionDataset):
    def __init__(
        self,
        dataset_dir: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        if dataset_dir is None:
            dataset_dir = "/private/home/rdessi/ConceptualCaptions"
        super(ConceptualCaptionsDataset, self).__init__(
            dataset_dir, transform=transform
        )
        self.dataset_dir = Path(dataset_dir)
        assert split in ["train", "test"]

        if split == "train":
            annotations_file = "train_conceptual_captions_paths.txt"
            self.image_folder = self.dataset_dir / "training"
            self.captions = None
        else:
            annotations_file = "test_conceptual_captions_paths.txt"
            self.image_folder = self.dataset_dir / "validation"
            with open(self.dataset_dir / "human_test_captions_reference.txt") as f:
                self.captions = f.readlines()

        self.samples = []
        with open(self.dataset_dir / annotations_file) as f:
            self.samples = f.readlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname = self.samples[index]

        img_path = self.image_folder / fname.strip()

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        sender_input, recv_input = self.transform(image)

        aux = {"img_id": fname}
        if self.captions:
            try:
                # preventing IndexError when loading images from txt file of img paths
                # only 13k captions but 13084 images
                # this is not a problem if batching w/ bsz == 100 as we usually do
                aux["captions"] = self.captions[index].strip()
            except IndexError:
                aux["captions"] = ""

        return sender_input, torch.tensor([index]), recv_input, aux


class ConceptualCaptionsWrapper:
    def __init__(self, dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = "/private/home/rdessi/ConceptualCaptions"
        self.dataset_dir = Path(dataset_dir)

    def get_split(
        self,
        split: str,
        batch_size: int,
        transform: Callable,
        num_workers: int = 1,
        shuffle: bool = None,
        seed: int = 111,
    ):
        ds = ConceptualCaptionsDataset(
            dataset_dir=self.dataset_dir, split=split, transform=transform
        )

        print(f"| Split {split} has {len(ds)} elements.")

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


if __name__ == "__main__":
    import clip

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()

    dataset_dir = "/private/home/rdessi/ConceptualCaptions"
    wrapper = ConceptualCaptionsWrapper(dataset_dir)
    dl = wrapper.get_split(
        split="train",
        batch_size=10,
        image_size=224,
        shuffle=False,
        num_workers=8,
    )

    nns = torch.load(
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/train_conceptual.nns.pt"
    )
    batch_emb = torch.load(
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/train_conceptual.emb.pt"
    ).to("cuda")
    clip = clip.load("ViT-B/32")[0]
    convert_models_to_fp32(clip)
    clip.eval()

    for i, elem in enumerate(dl):
        s_inp, labels, r_in, aux = elem
        feats = clip.encode_image(s_inp.to("cuda"))
        feats /= feats.norm(dim=-1, keepdim=True)
        breakpoint()
