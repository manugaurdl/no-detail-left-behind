import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import json

from egg.zoo.emergent_captioner.dataloaders import get_transform
from transformers import GPT2Tokenizer


clipcap_transform = get_transform(224, None)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

max_len = 50


def get_H(model, images, token_list):

    """
    passing 1 image with 2 caps
    """
    all_tokens = []
    all_masks = []
    for tokens in token_list:  
        padding = max_len - len(tokens)
        pad = torch.zeros(padding)
        mask = torch.cat((torch.ones(10+len(tokens)), pad)).cuda()
        tokens = torch.cat((tokens, pad)).int()
        all_tokens.append(tokens)
        all_masks.append(mask)
    tokens = torch.stack(all_tokens, dim = 0)
    mask = torch.stack(all_masks, dim = 0)
    images = [image.to(next(iter(model.clipcap.clip_project.parameters())).device) for image in images]
    tokens = tokens.to(next(iter(model.clipcap.clip_project.parameters())).device)
    image_feats = [model.clip.visual(image.unsqueeze(0)) for image in images]  # 1,512
    image_feats = torch.stack(image_feats, dim =0).squeeze(1)
    prompts = model.clipcap.clip_project(image_feats)
    prompts = prompts.view(image_feats.shape[0], 10, -1) # 1,10,768
    bsz, prefix_len, h_dim = prompts.shape 
    tokens_flat = tokens.view(-1,tokens.shape[-1]) # 2, 30
    token_emb = model.clipcap.gpt.transformer.wte(tokens_flat) #2,40,768
    gpt_input = torch.cat((prompts, token_emb), dim = 1) # 2,40,768
    mask = mask.view(-1, mask.shape[-1]) # 2,40
    out = model.clipcap.gpt(inputs_embeds = gpt_input, attention_mask = mask)
    probs = out.logits[:, 9: -1, :len(gpt2_tokenizer)].softmax(dim=-1).cpu()[: , :, :] #2,30,vocab
    
    log_probs = torch.log(probs)
    H = (probs * log_probs).sum(-1)
    mask = mask[:, 10:].cpu()
    H = (H * mask).sum(-1)
    all_len = mask.sum(-1)
    H = -1*(H /all_len)
    return H

    # probs = torch.gather(probs, dim = -1, index = tokens.unsqueeze(2).to(torch.int64).cpu()).squeeze(-1)
    
    # probs*= mask[:, 10:].cpu()
    # return probs





def calc_entropy(model, preds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    ids = list(json.load(open("/home/manugaur/img_cap_self_retrieval/data/parse_coco_req/test_cocoids.json", "r"))) 
    bsz = 100
    iters = len(ids)//bsz

    for i in tqdm(range(iters), total = iters):

        batch = ids[i*bsz : i*bsz + bsz]
        imgs = [clipcap_transform(Image.open(f"/home/manugaur/coco/val2014/COCO_val2014_{int(cocoid):012d}.jpg").convert("RGB")) for cocoid in batch]
        caps = [torch.tensor(gpt2_tokenizer.encode(preds[cocoid]),dtype=torch.int) for cocoid in batch]
        
        H = get_H(model, imgs, caps)

        # probs = probs[probs!=0]
        out.update(dict(zip(batch, [entropy.item() for entropy in H])))
    


    return out


