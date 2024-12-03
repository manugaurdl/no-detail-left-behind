import clip
import pickle
import sys
import time
import os
import json 
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import get_linear_schedule_with_warmup
# from egg.zoo.emergent_captioner.finetuning.utils import get_config, process_config, get_cl_args, init_wandb, get_best_state_dict, int2mil
from egg.zoo.emergent_captioner.finetuning.losses import get_loss, DiscriminativeLoss

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

"""
data : misc_data : cocoid2didx, img_feats, inference_preds
params : config 

"""
base_dir = "/home/manugaur"

def eval_on_bags(config):


    config['split'] = "test_val"
    config['scorer'] =  "vitb32"
    captioner = f"{config['captions_type']}_{config['opts']['checkpoint_dir'].split('/')[-1]}.pkl"        

    config["opts"]["batch_size"]= 100
    print(f"| Evaluating on benchmark for {config['captions_type']} trained model")
    
    data_dir = os.path.join(base_dir, "nips_benchmark/")

    with open(os.path.join(base_dir, "nips_benchmark/misc_data", f"coco_{config['split']}_cocoid2idx.json"), "r") as f:
        cocoid2idx = json.load(f)
    
    idx2cocoid = {v : k for k, v in cocoid2idx.items()}

    img_feats = torch.load(os.path.join(data_dir, "img_feats", f"coco_{config['split']}_{config['scorer']}.pt"))
    # sender_input, aux : cocoid, captions
    recall_1 = []
    recall_5 = []
    clip_s = []
    mean_rank = []
    median_rank = []
    loss = DiscriminativeLoss()
    
    with torch.no_grad():
        model_name = "ViT-B/32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        preds_path = os.path.join(base_dir, f"EGG/inference_preds/{captioner}")

        with open(preds_path, "rb") as f:
            preds = pickle.load(f)
        
        get_acc_5 = False

        for bag_size in [3,5,7,10]: 
            bag_dir =os.path.join(base_dir, "nips_benchmark/benchmark/benchmark/final_benchmark")
            
            #benchmark : list of bags. Each bag: list of cocoids    
            with open(os.path.join(bag_dir, f"{bag_size}.json"), "r") as f:
                listofbags = json.load(f)

            #OLD BAGS
            # with open(os.path.join("/home/manugaur/nips_benchmark/bags/clip_vitl14_mm_holistic", f"bsz_{bag_size}_thresh_{threshold}.json"), "r") as f:
            #     listofbags = json.load(f)
            # num_bags = 100
            # threshold = 0

            benchmark = []
            for bag in listofbags:
                benchmark.append([cocoid2idx[str(cocoid)] for cocoid in bag])
            
            for idx , bag in tqdm(enumerate(benchmark), total = len(benchmark)):
                    

                bag_img_feats = img_feats[bag]
                batch_caps = [preds[int(idx2cocoid[_])].strip() for _ in bag]

                with torch.no_grad():
                    batch_text_feats = model.encode_text(clip.tokenize(batch_caps, context_length=77, truncate=True).to(device))
                    batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim=True)

                    bag_img_feats = bag_img_feats / bag_img_feats.norm(dim=-1, keepdim = True)
                    batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim = True)
                    

                    _, acc = loss(batch_text_feats,bag_img_feats, training=False, get_acc_5 = False, aux_input=None)
                    recall_1.append([_.item() for _ in acc['acc']])
                    clip_s.append(acc['clip_s'].mean().item())         

            print(f"| BAG SIZE = {bag_size}")      
            
            save_dir = os.path.join(base_dir, f"nips_benchmark/recall_per_bag/final")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            with open(os.path.join(save_dir,f"bsz_{bag_size}_{captioner}.json"), "w") as f:
                json.dump(recall_1, f)

            print(f"Recall@1 : {round(np.array(recall_1).mean()*100,1)}")
            print(f"CLIP score : {round(np.array(clip_s).mean(), 2):.2f}")
            recall_1 = []

