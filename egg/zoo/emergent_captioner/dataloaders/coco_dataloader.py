# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable
import pickle
import time
import torch
import json
import torch.distributed as dist
from PIL import Image
from transformers import GPT2Tokenizer
from tqdm import tqdm
from egg.zoo.emergent_captioner.dataloaders.utils import MyDistributedSampler, ValSampler
from torch.utils.data.distributed import DistributedSampler
# from llava.mm_utils import process_images
from transformers import CLIPImageProcessor

# from torchvision import transforms
# TRANSFORM = transforms.Compose([
#     transforms.ToTensor()
# ])

def open_pickle(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file 

class CocoDataset:
    def __init__(self, root, samples, mle_train, split, caps_per_img, captions_type, max_len_token, prefix_len, transform, debug, mllm):
        self.root = root
        self.samples = samples
        self.transform = transform
        self.debug = debug
        self.split = split
        self.mle_train = mle_train
        self.max_len_token = max_len_token
        self.prefix_len = prefix_len
        self.captions_type = captions_type
        self.mllm = mllm

        if self.mle_train:
            self.path2tokens = os.path.join(root, f"tokenized_caps/{self.captions_type}/{self.split}")
            # self.id2tokens = torch.load
            pass
        self.caps_per_img = caps_per_img
        self.lazy_feat_dir = "/ssd_scratch/cvit/manu/EGG/lazy_clip_feats"

    def __len__(self):
        if self.debug:
            return 200
        else:
            return len(self.samples)
    
    def pad(self,tokens):
                     
        padding = self.max_len_token - tokens.shape[-1]

        if padding>0:
            pad = torch.zeros(padding)
            pad = pad.masked_fill(pad ==0, -1) 
            tokens = torch.cat((tokens, pad)).int() # tokens is padded with -1.

            # ### padded tokens replace the tokens. Here the padding is done by -1. But the tokens returned by the method have padding with 0.
            # if not self.lazy_load:
            #     self.tokenized_captions[idx][cap_idx] = tokens
        else:
            # if caption > max_len, truncate it 
            tokens = tokens[:self.max_len_token]
            # if not self.lazy_load:
            #     self.tokenized_captions[idx][cap_idx] = tokens
            
        mask = tokens.ge(0) #True for indices > 0 i,e padded indices = False
        tokens[~mask] =0  # padding now done with 0
        mask = torch.cat((torch.ones(self.prefix_len),mask)) 
        
        return (tokens, mask)


    def get_tokens(self,cocoid):
        path =  f"{self.path2tokens}/{cocoid}"
        tokens = []
        masks = []
        for i in range(self.caps_per_img):
            t, m = self.pad(torch.load(path + f"_{i}.pt"))
            tokens.append(t)
            masks.append(m)

        return torch.stack(tokens), torch.stack(masks)

    def __getitem__(self, idx):
        file_path, captions, image_id = self.samples[idx]

        # # If load CLIP to GPU
        image = Image.open(os.path.join(self.root, file_path)).convert("RGB")
        
        if self.mllm == "clipcap":
            sender_input = self.transform(image) # they are same
            recv_input = sender_input
            # torch.save(sender_input, os.path.join("/home/manugaur/EGG/sender_input", f"{image_id}.pt"))

        elif self.mllm=="llava":
            _ , recv_input = self.transform(image)
            image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
            llava_mistral_config = pickle.load(open("/home/manugaur/EGG/llava_mistral_config.pkl", "rb"))

            image_sizes = image.size
            image = image.resize((640,480))
            sender_input = process_images(
                [image],
                image_processor,
                llava_mistral_config).squeeze(0).to(dtype=torch.float16)

        elif self.mllm=="llava-phi":
            _ , recv_input = self.transform(image)
            # image = image.resize((640,480))
            # sender_input = TRANSFORM(image)

        ## Lazy loading
        # sender_input = torch.load(os.path.join(self.lazy_feat_dir, f"{image_id}.pt"), map_location="cpu")#.float()

        """ Saving sender_inputs"""
        # save_dir = f"/home/manugaur/EGG/sender_inputs/{self.split}"
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)
        # torch.save(sender_input, os.path.join(save_dir, f"{image_id}.pt"))

        if self.mle_train:
            padded_tokens, mask = self.get_tokens(image_id)
            aux = {"cocoid": torch.tensor([image_id]), "captions": captions[:self.caps_per_img], "tokens": padded_tokens, "mask" : mask}
        else:
            aux = {"cocoid": torch.tensor([image_id]), "captions": captions[:self.caps_per_img]}

        if self.mllm=='llava-phi':
            sender_input = recv_input

        return sender_input, torch.tensor(image_id), recv_input, aux

class CocoNegDataset:
    def __init__(self, root, samples, mle_train, split, caps_per_img, captions_type, max_len_token, prefix_len, transform, debug, bags, cocoid2samples_idx, mllm):
        self.root = root
        self.samples = samples
        self.transform = transform
        self.debug = debug
        self.split = split
        self.mle_train = mle_train
        self.max_len_token = max_len_token
        self.prefix_len = prefix_len
        self.captions_type = captions_type
        # if self.mle_train:
        #     self.path2tokens = os.path.join(root, f"tokenized_caps/{self.captions_type}/{self.split}")
            # self.id2tokens = torch.load
            # pass
        self.caps_per_img = caps_per_img
        
        self.cocoid2samples_idx = cocoid2samples_idx[split]
        self.bags = bags
        self.mllm = mllm
        self.a100_dir =  "/home/ubuntu/pranav/pick_edit"
    def __len__(self):
        if self.debug:
            return 50
        else:
            return len(self.bags)


    def __getitem__(self, idx):

        bag = self.bags[idx]
        sender_inputs = []
        aux_list = []
        cocoids =[]
        bag_of_caps = []
        if self.mllm =="clipcap":
            for cocoid in bag:
                sample_idx = self.cocoid2samples_idx[cocoid]
                file_path, captions, image_id = self.samples[sample_idx]
                assert image_id == cocoid
                
                if os.path.isdir(self.a100_dir):
#                    torch.save(self.transform(Image.open(os.path.join(self.root, file_path)).convert("RGB")), os.path.join(a100_dir, f"{cocoid}.pt"))
                    sender_inputs.append(torch.load(os.path.join("/mnt/localdisk-1/figma_scrapped_data/pick_edit", f"{cocoid}.pt")))
                else:
                    sender_inputs.append(torch.load(os.path.join(f"/home/manugaur/EGG/sender_inputs/", f"{cocoid}.pt")))

                # sender_inputs.append(self.transform(Image.open(os.path.join(self.root, file_path)).convert("RGB")))

                # aux_list.append({"cocoid": torch.tensor([image_id]), "captions": captions[:5]}) 
                # aux_list.append({"captions": captions[:5]}) 
                cocoids.append(cocoid)
                bag_of_caps.append(captions[:self.caps_per_img])
            # sender_input = torch.stack(sender_inputs)
        

        #if scorer = CLIP ViT-L : Just pass cocoid. Llava will preproc image. Feed that preproc_img to receiver.
        #if scorer = CLIP ViT-B : PIL image fed to LLAVA_preproc and CLIP_ViTB_transform separately.
        
        elif self.mllm =="llava-phi":
            for cocoid in bag:
                sample_idx = self.cocoid2samples_idx[cocoid]
                file_path, captions, image_id = self.samples[sample_idx]
                assert image_id == cocoid
                sender_inputs = None  
                cocoids.append(cocoid)
                bag_of_caps.append(captions[:self.caps_per_img])


        return sender_inputs, cocoids, bag_of_caps

def hard_neg_collate(og_batch):
    cocoids = []
    sender_input = []
    all_bags_caps = [] 

    for i in og_batch:
        if i[0] is not None:
            sender_input.extend(i[0]) 
        cocoids.extend(i[1])
        all_bags_caps.extend(i[-1])
    if len(sender_input)>0:
        sender_input = torch.stack(sender_input)
    cocoids = torch.tensor(cocoids)

    # caps zipped and changed from 5 X 100 --> 100 X 5
    aux = {"cocoid" : cocoids, 
            "captions" : [list(caps_per_capidx) for caps_per_capidx in zip(*all_bags_caps)]}
    
    # batch : sender_inp, sample_idxs, receiver input, aux_dict  dds
    return sender_input, cocoids, sender_input, aux

class CocoWrapper:

    def __init__(self, captions_type : str,  dataset_dir: str, neg_mining : dict):
        self.num_omitted_ids = 0
        self.dataset_dir = Path(dataset_dir)
        self.captions_type = captions_type
        self.neg_mining = neg_mining

        if self.captions_type != "coco":
            self.id2caption = open_pickle(os.path.join(dataset_dir, f"synthetic_data/{self.captions_type}_preproc_5.pkl"))
            assert isinstance(list(self.id2caption.values())[0], list), "cocoid2cap is not id --> list of caps"
        
        self.split2samples = self._load_splits(self.dataset_dir) # {test,val,train,restval} --> {test[0] :(img_path, list of 5 caps, cocoid)}
        self.cocoid2samples_idx = self.get_cocoid2sample_idx()   # cocoid <--> dataset idx         

        # val_test_list = self.split2samples['test']
        # val_test_list.extend(self.split2samples['val'])
        # self.split2samples['test'] = val_test_list
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # neg_train = False
        # if any(key in self.neg_mining["curricullum"] for key in ['easy', "medium", "hard"]):
        #     neg_train = True
        self.split2bags = self.load_bags(jatayu, neg_mining['curricullum'])
        print(f"{self.num_omitted_ids} cocoids are removed during preproc for {self.captions_type} captions")

    def load_bags(self, jatayu, curricullum):
        if jatayu:
            # path2bags = "/home/manugaur/EGG/hard_negs/bags/top_k_sim/"  
            path2bags = "/home/manugaur/EGG/hard_negs/bags/diff_levels/"
        elif os.path.isdir("/home/ubuntu/pranav/pick_edit"):
            path2bags = "/home/ubuntu/pranav/pick_edit/EGG/hard_negs/bags/diff_levels/"
        else:
            path2bags = "/ssd_scratch/cvit/manu/EGG/hard_negs/bags/top_k_sim/"

        """"split  = "train" | Removed val and testing. Train on this. Eval on our bags. """
        
        split2bags = {}
        for i in list(curricullum.values()):
            level, bsz = i 
            if level =="rand":
                continue
            if level=="rand_crrclm":
                l = []
                for diff in ['easy', 'medium', 'train']:
                    l.extend(pickle.load(open(os.path.join(path2bags,f"{diff}/train/bsz_{bsz}.pkl"), "rb")))
            else:
                l = pickle.load(open(os.path.join(path2bags,f"{level}/train/bsz_{bsz}.pkl"), "rb"))
            split2bags[f"{level}_{bsz}"] = l
        
        
        # for level in diff_levels:

        #     if split in ['val', "test"]:
        #         if level in ["medium", "hard"]:
        #             with open(os.path.join(path2bags,level, split, f"bsz_{self.neg_mining['val_bag_size']}.pkl"), "rb") as f:
        #                 split2bags[level][split] = pickle.load(f)
        #         else:
        #             continue
        #     else:
        #         with open(os.path.join(path2bags,level, split, f"bsz_{self.neg_mining['bag_size']}.pkl"), "rb") as f:
        #             split2bags[level][split] = pickle.load(f)
        
        return split2bags

    def tokenize(self,split):
        """
        self.split2samples[split] : list of [img_path, list_of_caps, cocoid]
        """

        self.all_len = []
        save_dir = os.path.join(self.dataset_dir, f"tokenized_caps/{self.captions_type}/{split}/")
        if not os.path.isdir(save_dir) or len(os.listdir(save_dir))<1000:
            print(f"tokenizing {split} captions...")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)    
            for instance in tqdm(self.split2samples[split], total = len(self.split2samples[split])):
                cocoid = instance[2]
                captions = instance[1]
                if isinstance(captions, tuple) or isinstance(captions, list):
                    for idx, cap in enumerate(captions):
                        token = torch.tensor(self.tokenizer.encode(cap),dtype=torch.int)
                        torch.save(token, os.path.join(save_dir, f"{cocoid}_{idx}.pt"))
                    # tokens = [torch.tensor(self.tokenizer.encode(cap),dtype=torch.int) for cap in caption]
                        self.all_len.append(token.shape[-1])
            with open(os.path.join(self.dataset_dir, f"tokenized_caps/{self.captions_type}/{split}/all_len.json"), "w") as f:
                json.dump(self.all_len, f)

        else:
            print(print(f"tokenized {split} captions exist"))

    def _load_splits(self, dataset_dir):
        path2ann = os.path.join(dataset_dir, "annotations/dataset_coco.json")

        with open(path2ann) as f:
            annotations = json.load(f)
        split2samples = defaultdict(list)
        
        for img_ann in annotations["images"]:

            file_path = self.dataset_dir / img_ann["filepath"] / img_ann["filename"]
            cocoid = img_ann["cocoid"]
            try:
                if self.captions_type =="coco":
                    captions = [x["raw"] for x in img_ann["sentences"]]
                else:
                    captions = self.id2caption[cocoid]
            except KeyError:
                self.num_omitted_ids+=1
            # img_id = img_ann["imgid"]
            split = img_ann["split"]

            split2samples[split].append((file_path, captions, cocoid))
        if "restval" in split2samples:
            split2samples["train"] += split2samples["restval"]

        # for k, v in split2samples.items():
        #     print(f"| Split {k} has {len(v)} elements.")
        return split2samples

    def get_cocoid2sample_idx(self):
        cocoid2samples_idx = {}
        for split in list(self.split2samples.keys()):
            cocoid2idx = {}
            for idx, sample in enumerate(self.split2samples[split]):
                cocoid2idx[sample[-1]] = idx
            cocoid2samples_idx[split] = cocoid2idx
        return cocoid2samples_idx

    def get_split(
        self,
        split: str,
        caps_per_img : int,
        neg_mining : bool,
        debug : bool,
        batch_size: int,
        mle_train : bool,
        max_len_token : int,
        prefix_len : int,
        is_dist_leader : bool,
        transform: Callable,
        mllm : str,
        num_workers: int = 8,
        seed: int = 111,
        level_bsz : str = None,

    ):
        if level_bsz is not None:
            level, bsz = level_bsz.split("_")
        if mle_train and split != "test":
            self.tokenize(split)
        shuffle = not debug and split == "train"
        if split == "test_val":
            samples = self.split2samples["test"]
            samples.extend(self.split2samples["val"])
        else: 
            samples = self.split2samples[split]
        assert samples, f"Wrong split {split}"
       
        if neg_mining:
            if level == 'rand_crrclm':
                assert split == 'train', Exception("Random Crrclm is only done for train set.")
            bags = self.split2bags[level_bsz] # samples in bags are in cocoid format.
            ds = CocoNegDataset(self.dataset_dir, samples, mle_train, split, caps_per_img, self.captions_type, max_len_token, prefix_len, transform, debug, bags, self.cocoid2samples_idx, mllm = mllm)
        else :
            ds = CocoDataset(self.dataset_dir, samples, mle_train, split, caps_per_img, self.captions_type, max_len_token, prefix_len,transform=transform, debug = debug, mllm = mllm)
        sampler = None

        if dist.is_initialized() and split =="train":
            print(f"{split} data is distributed.")
            if shuffle is None:
                shuffle = split != "test"

            # sampler = MyDistributedSampler(
            #     ds, shuffle=shuffle, drop_last=True, seed=seed
            # )
            sampler = DistributedSampler(ds, num_replicas=int(os.environ["LOCAL_WORLD_SIZE"]), rank= int(os.environ["LOCAL_RANK"]), shuffle=True, drop_last=True)

        if split in ["val", "test"]:
            if neg_mining:
                sampler = ValSampler(ds, debug, self.neg_mining["val_bag_size"])
            else:
                sampler = ValSampler(ds, debug, None)

        if sampler is not None :
            shuffle=None

        if split=="eval":
            shuffle = True
            sampler = None

        if neg_mining:
            bag_size = int(self.neg_mining["val_bag_size"] if split == "val" else bsz)
            bags_per_batch = int(batch_size/bag_size)

            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=bags_per_batch,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn = hard_neg_collate,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
                
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
    wrapper = CocoWrapper()
    dl = wrapper.get_split(
        split="test",
        batch_size=10,
        image_size=224,
        shuffle=False,
        num_workers=8,
    )

    for i, elem in enumerate(dl):
        breakpoint()

