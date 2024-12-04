import clip
import pickle
import sys
from PIL import Image
import time
import os
import json 
import torch
import numpy as np
from tqdm import tqdm
import random
import argparse
import egg.core as core 
from egg.zoo.emergent_captioner.finetuning.utils import get_config, process_config, get_cl_args
from egg.zoo.emergent_captioner.finetuning.losses import DiscriminativeLoss
from egg.zoo.emergent_captioner.finetuning.game import build_game
from egg.zoo.emergent_captioner.finetuning.opts import get_common_opts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cocoid2img(coco_path, cocoid):
    img_path = os.path.join(coco_path, f"COCO_train2014_{int(cocoid):012d}.jpg")
    if not os.path.isfile(img_path):
        return os.path.join(coco_path, f"COCO_val2014_{int(cocoid):012d}.jpg")
    return img_path

def load_truematch_bags(bag_dir, bag_size, cocoid2idx): 
    """list of list of cococids"""
    listofbags = json.load(open(os.path.join(bag_dir, f"{bag_size}.json"), "r")) 
    out = []
    for bag in listofbags:
        out.append([cocoid2idx[str(cocoid)] for cocoid in bag])
    return out

def load_scorer():
    model_name = "ViT-B/32"
    model, preprocess = clip.load(model_name, device=DEVICE)
    model.eval()
    return model, preprocess


def main(parser):
    
    start = time.time()
    os.makedirs(args.out_dir, exist_ok = True)
    cocoid2idx = json.load(open("./data/data/test_val_cocoid2idx.json", "r"))
    idx2cocoid = {v : k for k, v in cocoid2idx.items()}
    loss = DiscriminativeLoss()
    model, preprocess = load_scorer()
        
    with open(args.preds_path, "rb") as f:
        preds = pickle.load(f)

    def encode(bag):
        caps = [preds[int(idx2cocoid[_])].strip() for _ in bag]
        images = torch.cat([preprocess(Image.open(cocoid2img("./data/data/truematch_images", int(idx2cocoid[_])))).unsqueeze(0).to(DEVICE) for _ in bag])
        text_feat = model.encode_text(clip.tokenize(caps, context_length=77, truncate=True).to(DEVICE))
        img_feat = model.encode_image(images)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim = True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim = True)

        return text_feat, img_feat
    
    recall_per_bag = []
    
    for bag_size in [3,5,7]:
        recall_1 = []
        clip_s = []
        truematch_bags = load_truematch_bags("./data/data/benchmark", bag_size, cocoid2idx)
        
        for idx , bag in tqdm(enumerate(truematch_bags), total = len(truematch_bags)):
            
            with torch.no_grad():
                text_feat, img_feat = encode(bag)                
                _, acc = loss(text_feat,img_feat, training=False, get_acc_5 = False, aux_input=None)
                recall_1.append([_.item() for _ in acc['acc']])
                clip_s.append(acc['clip_s'].mean().item())         
                recall_per_bag.append([i.item() for i in acc['acc']])
        
        print(f"| BAG SIZE = {bag_size}")
        print(f"Recall@1 : {round(np.array(recall_1).mean()*100,1)}")
        print(f"CLIP score : {round(np.array(clip_s).mean(), 2):.2f}")

    #save R@1 for each image
    with open(os.path.join(args.out_dir, f"{args.preds_path.split('/')[-1].split('.')[0]}.json"), "w") as f:
        json.dump(recall_per_bag, f)
    
    end = time.time()
    print(f" \n| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path", help = "path to captions")
    parser.add_argument("--out_dir", help= "output dir")
    args = parser.parse_args()
    main(parser)



