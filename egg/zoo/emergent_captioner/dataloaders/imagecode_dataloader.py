# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Callable
from PIL import Image

import torch
import torch.distributed as dist


from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler


def collate(batch):
    """
    list of samples : 
    sample[0] --> sender_input (1, 3, H,W)
    sample[1] --> dataset iddx
    sample[2] --> recv_input (10,2 ,H, W)
    """

    recv_inputs = []
    ids = []
    caps = []
    for sample in batch:
        recv_inputs.append(sample[2])
        ids.append(sample[1])
        caps.append(sample[-1]['captions'])

    recv_inputs = torch.stack(recv_inputs)
    sender_inputs = recv_inputs[:, 0]
    ids = torch.stack(ids)

    return sender_inputs, ids, recv_inputs, {"captions" : caps} 
    
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, bool) or isinstance(elem, str):
        return batch
    elif isinstance(elem, dict):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, list) or isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise RuntimeError("Cannot collate batch")


class ImageCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        image_dir,
        split,
        transform,
    ):
        assert split in ["train", "valid", "test"]

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        with open(dataset_dir / f"{split}_data.json", "r") as fd:
            data = json.load(fd)

        self.samples = []
        for img_dir, sents in data.items():
            for img_idx, text in sents.items():
                self.samples.append((img_dir, int(img_idx), text))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        set_dir, img_idx, text = self.samples[idx]  # image set | 
        set_path = Path(self.image_dir) / set_dir
        img_files = list(set_path.glob("*.jpg"))
        img_files.sort(key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:]))

        images = [Image.open(photo_file) for photo_file in img_files]
        images[0], images[img_idx] = images[img_idx], images[0] #Image to be self-retrieved placed in idx=0

        sender_imgs = [self.transform(photo) for photo in images]

        sender_input = torch.stack(sender_imgs)[torch.LongTensor([0])]  #image to generate caption for 
        recv_input = torch.stack(sender_imgs) # bag of 10 images (10,3,3,224)

        aux_input = {
            "captions": text,
            "is_video": torch.Tensor(["open-images" not in set_dir]),
            "target_idx_imagecode": torch.tensor([img_idx]).long(),
        }
        return sender_input, torch.tensor([idx]), recv_input, aux_input


class ImageCodeWrapper:
    def __init__(self, captions_type : str,  dataset_dir: str, jatayu: bool, neg_mining : dict, image_dir : str):
        if dataset_dir is None:
            dataset_dir = "/workspace/manugaur/image_code"
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = Path(image_dir)

    def get_split(
        self,
        split: str,
        transform: Callable,
        batch_size: int = 1,
        num_workers: int = 8,
        shuffle: bool = None,
        seed: int = 111,
    ):
        ds = ImageCodeDataset(self.dataset_dir, self.image_dir,  split=split, transform=transform)

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
            batch_size=batch_size,  # image sets are already batch in groups of 10
            shuffle=sampler is None and split != "test",
            sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        return loader


if __name__ == "__main__":
    w = ImageCodeWrapper()
    dl = w.get_split(split="test", num_workers=0)
    for i, elem in enumerate(dl):
        if i == 10:
            break
        continue
