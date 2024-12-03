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
from egg.zoo.emergent_captioner.finetuning.utils import get_config, process_config, get_cl_args, init_wandb, get_best_state_dict, int2mil
from egg.zoo.emergent_captioner.finetuning.losses import get_loss, DiscriminativeLoss

import egg.core as core
from egg.core import ConsoleLogger
from egg.zoo.emergent_captioner.dataloaders import (
    CocoWrapper,
    ConceptualCaptionsWrapper,
    FlickrWrapper,
    get_transform,
)
from egg.zoo.emergent_captioner.finetuning.game import build_game
from egg.zoo.emergent_captioner.finetuning.opts import get_common_opts
from egg.zoo.emergent_captioner.finetuning.utils import ModelSaver
from egg.zoo.emergent_captioner.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    print_grad_info,
    setup_for_distributed,
    store_job_and_task_id,
)

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"
# os.environ["CUDA_VISIBLE_DEVICES"] = str((0))


def main(params, config):
    if config['baseline'] is not None:
        assert(config['use_benchmark'] and not config["use_gt"], "use_gt must be False and use_benchmark must be True when baseline is not None")

    print(f"Self Retrieval using {config['captions_type']} captions")
    start = time.time()
    opts = get_common_opts(params=params)
    opts.jatayu = os.path.isdir("/home/manugaur")
    # opts.loss_type= config['train_method']

    print(get_sha())
    
    name2wrapper = {
        "conceptual": ConceptualCaptionsWrapper,
        "coco": CocoWrapper,
        "flickr": FlickrWrapper,
    }
    # args
    config["neg_mining"]["do"] = False
    wrapper = name2wrapper[opts.train_dataset](captions_type = config["captions_type"], dataset_dir = opts.dataset_dir, jatayu = opts.jatayu, neg_mining = config["neg_mining"])
    
    data_kwargs = dict(
        batch_size=config["opts"]["batch_size"],
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        mllm = "clipcap",
        num_workers=config["num_workers"],
        seed=opts.random_seed,
        debug = False,
        mle_train = config["train_method"] =="mle",
        max_len_token = opts.max_len,
        prefix_len = config["prefix_len"],
        is_dist_leader = opts.distributed_context.is_leader,
    )

    test_loader = wrapper.get_split(split=config['split'], caps_per_img = 5, neg_mining = False, **data_kwargs)
    
    data_dir = "/home/manugaur/nips_benchmark/"

    with open(os.path.join("/home/manugaur/nips_benchmark/misc_data", f"coco_{config['split']}_cocoid2idx.json"), "r") as f:
        cocoid2idx = json.load(f)
    idx2cocoid = {v : k for k, v in cocoid2idx.items()}
    # GT image and caption CLIP feats
    # img_feats = torch.load(os.path.join(data_dir, "img_feats", f"coco_test_vitl14.pt"))
    # text_feats = torch.load(os.path.join(data_dir, "text_feats", f"{config['captions_type']}_test_vitl14.pt"))
    if config['use_gt']:
        if config['avg_text_feat']:
            text_feat_name = f"{config['data']}_{config['split']}_{config['scorer']}_avg.pt"
        else:
            text_feat_name = f"{config['data']}_{config['split']}_{config['scorer']}.pt"
        text_feats = torch.load(os.path.join(data_dir, "text_feats", text_feat_name))
    img_feats = torch.load(os.path.join(data_dir, "img_feats", f"coco_{config['split']}_{config['scorer']}.pt"))
    # sender_input, aux : cocoid, captions
    recall_1 = []
    recall_5 = []
    clip_s = []
    mean_rank = []
    median_rank = []
    loss = DiscriminativeLoss()
    if config['use_greedy'] and config['avg_text_feat']:
        raise Exception("Can't have greedy and avg_text_feat together")
    if config['use_benchmark'] and config['split']!="test_val":
        raise Exception("Split should be test_val when evaluating on bags")
#------------------------------------------------------------------------------------------------------------------------------------------------
    """RANDOM 100 batch_size batch using GT"""
    
    if not config['use_benchmark'] and config['use_gt']:
        for batch in tqdm(test_loader, total = len(test_loader)):
            _,_,_, aux = batch
            clip_idx = [cocoid2idx[str(cocoid.item())] for cocoid in aux["cocoid"]]

            batch_img_feats = img_feats[clip_idx]
            batch_text_feats = text_feats[clip_idx]
            batch_img_feats = batch_img_feats / batch_img_feats.norm(dim=-1, keepdim = True)
            batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim = True)

            acc_per_cap = []
            acc_5_per_cap =[]
            clip_s_per_cap = []
            if config['avg_text_feat']:
                _, acc = loss(batch_text_feats.squeeze(1),batch_img_feats, False, True, None)
                recall_1.append(acc['acc'].mean().item())
                recall_5.append(acc['acc_5'].mean().item())
                clip_s.append(acc['clip_s'].mean().item())
            else:
                if config['use_greedy']:
                    if config['data']=="coco":
                        raise Exception("No greedy for COCO")

                    _, acc = loss(batch_text_feats[:,0,:],batch_img_feats, False, True, None)
                    recall_1.append(acc['acc'].mean().item())
                    recall_5.append(acc['acc_5'].mean().item())
                    clip_s.append(acc['clip_s'].mean().item())
                    
                else:
                    for i in range(batch_text_feats.shape[1]):
                        _, acc = loss(batch_text_feats[:,i,:],batch_img_feats, False, True, None)
                        acc_per_cap.append(acc['acc'])
                        acc_5_per_cap.append(acc['acc_5'])
                        clip_s_per_cap.append(acc['clip_s'])

                    recall_1.append(torch.stack(acc_per_cap).mean(axis = 0).mean().item())
                    recall_5.append(torch.stack(acc_5_per_cap).mean(axis = 0).mean().item())
                    clip_s.append(torch.stack(clip_s_per_cap).mean(axis=0).mean().item())


        print(f"Recall@1 : {round(np.array(recall_1).mean()*100,1)}")
        print(f"Recall@5 : {round(np.array(recall_5).mean()*100,1)}")
        print(f"CLIP score : {round(np.array(clip_s).mean(), 1):.1f}")

#------------------------------------------------------------------------------------------------------------------------------------------------
    """RANDOM 100 batch_size batch using MODEL PREDS"""
    
    if not config['use_benchmark'] and not config['use_gt']:

        model_name = "ViT-B/32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        captioner = f"{config['data']}_{config['method']}"        
        preds_path = f"/home/manugaur/EGG/inference_preds/{captioner}.pkl"

        with open(preds_path, "rb") as f:
            preds = pickle.load(f)
        
        get_acc_5 = True

        for batch in tqdm(test_loader, total = len(test_loader)):
            _,_,_, aux = batch
            clip_idx = [cocoid2idx[str(cocoid.item())] for cocoid in aux["cocoid"]]

            batch_img_feats = img_feats[clip_idx]
            try:
                bag_caps  = [preds[_.item()] for _ in aux['cocoid']]
            except:
                print(True)
        
            with torch.no_grad():
                batch_text_feats = model.encode_text(clip.tokenize(bag_caps, context_length=77, truncate=True).to(device))
                batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim=True)
                
            batch_img_feats = batch_img_feats / batch_img_feats.norm(dim=-1, keepdim = True)
            batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim = True)

            _, acc = loss(batch_text_feats,batch_img_feats, False, True, None)
            recall_1.append(acc['acc'].mean().item())
            recall_5.append(acc['acc_5'].mean().item())
            clip_s.append(acc['clip_s'].mean().item())
                    

        print(f"Recall@1 : {round(np.array(recall_1).mean()*100,1)}")
        print(f"Recall@5 : {round(np.array(recall_5).mean()*100,1)}")
        print(f"CLIP score : {round(np.array(clip_s).mean(), 1):.1f}")


#------------------------------------------------------------------------------------------------------------------------------------------------
    # """RETRIEVAL WITHIN HARD BAGS using GT"""
    if config['use_benchmark'] and config['use_gt']:
        for bag_size in [3,5,7,10]: 
            bag_dir = "/home/manugaur/nips_benchmark/benchmark/benchmark/final_benchmark/"
            # bag_dir = "/home/manugaur/nips_benchmark/bags/clip_vitl14_mm_holistic" 

            if config['use_gt']:
                captioner = f"{config['data']}_gt"
            
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
                bag_text_feats = text_feats[bag]
                bag_img_feats = bag_img_feats / bag_img_feats.norm(dim=-1, keepdim = True)
                bag_text_feats = bag_text_feats / bag_text_feats.norm(dim=-1, keepdim = True)
                
                if config['use_greedy']:
                    bag_text_feats = bag_text_feats[:, 0, :]
                    _, acc = loss(bag_text_feats,bag_img_feats, training=False, get_acc_5 = False, aux_input=None)
                    recall_1.append([_.item() for _ in acc['acc']])
                    clip_s.append(acc['clip_s'].mean().item())
                
                else:
                    if config['avg_text_feat']:
                        _, acc = loss(bag_text_feats.squeeze(1),bag_img_feats, training=False, get_acc_5 = False, aux_input=None)
                        recall_1.append(acc['acc'].mean().item())
                        clip_s.append(acc['clip_s'].mean().item())                    
                    
                    else:
                        acc_per_cap = []
                        clip_s_per_cap = []
                        for i in range(bag_text_feats.shape[1]):
                            _, acc = loss(bag_text_feats[:,i,:],bag_img_feats, False, False, None)
                            acc_per_cap.append(acc['acc'].mean())
                            clip_s_per_cap.append(acc['clip_s'].mean())
                        recall_1.append(torch.stack(acc_per_cap).mean().item())
                        clip_s.append(torch.stack(clip_s_per_cap).mean().mean().item())
            print(f"| BAG SIZE = {bag_size}")      

            with open(f"/home/manugaur/nips_benchmark/recall_per_bag/final/bsz_{bag_size}_{captioner}.json", "w") as f:
                json.dump(recall_1, f)
            # print(f"{round(np.array(recall_1).mean()*100,2)}/ {np.array(mean_rank).mean():.2f}/ {np.array(median_rank).mean():.2f}")
            print(f"Recall@1 : {round(np.array(recall_1).mean()*100,1)}")
            # print(f"Mean rank : {np.array(mean_rank).mean():.2f}")
            # print(f"Median rank : {np.array(median_rank).mean():.2f}")
            print(f"CLIP score : {round(np.array(clip_s).mean(), 2):.2f}")
            recall_1 = []
#------------------------------------------------------------------------------------------------------------------------------------------------

    #"""RETRIEVAL WITHIN HARD BAGS using MODEL PREDS"""
    
    if config['use_benchmark'] and not config['use_gt']:

        model_name = "ViT-B/32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        if config['baseline'] is None:
            captioner = f"{config['data']}_{config['method']}"        
            preds_path = f"/home/manugaur/EGG/inference_preds/{captioner}.pkl"
            # preds_path = "/home/manugaur/nips_benchmark/baselines/instructblip2_vicuna7b_p2.pkl"

            with open(preds_path, "rb") as f:
                preds = pickle.load(f)
        else:
            preds = pickle.load(open(f"/home/manugaur/nips_benchmark/baselines/{config['baseline']}.pkl", 'rb'))
        get_acc_5 = False

        for bag_size in [3,5,7]: 
            bag_dir = "/home/manugaur/nips_benchmark/final_bags"
            
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

            # with open(f"/home/manugaur/nips_benchmark/recall_per_bag/final/bsz_{bag_size}_{captioner}.json", "w") as f:
            #     json.dump(recall_1, f)
            # print(f"{round(np.array(recall_1).mean()*100,2)}/ {np.array(mean_rank).mean():.2f}/ {np.array(median_rank).mean():.2f}")
            print(f"Recall@1 : {round(np.array(recall_1).mean()*100,1)}")
            # print(f"Mean rank : {np.array(mean_rank).mean():.2f}")
            # print(f"Median rank : {np.array(median_rank).mean():.2f}")
            print(f"CLIP score : {round(np.array(clip_s).mean(), 2):.2f}")
            recall_1 = []
#------------------------------------------------------------------------------------------------------------------------------------------------
    # if  RECALL_PER_BAG:
    #     with open(f"/home/manugaur/nips_benchmark/recall_per_bag/bsz_{bag_size}_thresh_{threshold}_{captioner}.json", "w") as f:
    #         json.dump(recall_1, f)
    # print(f"{round(np.array(recall_1).mean()*100,2)}/ {np.array(mean_rank).mean():.2f}/ {np.array(median_rank).mean():.2f}")
    # print(f"Recall@1 : {round(np.array(recall_1).mean()*100,2)}")
    # print(f"Recall@5 : {round(np.array(recall_5).mean()*100,2)}")
    # print(f"Mean rank : {np.array(mean_rank).mean():.2f}")
    # print(f"Median rank : {np.array(median_rank).mean():.2f}")
    # print(f"CLIP score : {round(np.array(clip_s).mean(), 2):.2f}")
    

        # batch_text_feats.view(batch_text_feats.shape[1], batch_text_feats.shape[0] , -1)[0]

    
    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    use_ddp = False    

    if "LOCAL_RANK" in os.environ:
        use_ddp = True
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config_filename = f"egg/zoo/emergent_captioner/finetuning/configs/{sys.argv[1:][0]}.yml"
    config = get_config(config_filename)
    config = process_config(config, use_ddp, sys.argv[1:])
    params = get_cl_args(config)
    
    # params
    config['use_benchmark'] = True
    config["use_gt"] = True
    config['baseline'] = "instblip_123k" # baseline can be MLLM baselines or GT caps with multiple caps per image
    config['method'] = None #"cider-sr_lambda_1_srlv_lr_1e7_g_baseline_curri"
    config['data'] = "blip2mistral"
    config["opts"]["batch_size"]= 100
    config['split'] = "val"
    if config['use_benchmark']:
        config['split'] = "test_val"
    config['scorer'] =  "vitb32"  # "", "vitb32"
    config['use_greedy'] = False
    config['avg_text_feat'] = False 
    main(params, config)