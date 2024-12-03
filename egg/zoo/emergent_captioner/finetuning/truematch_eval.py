import clip
from transformers import AutoProcessor, AutoModel
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
        print(f"Self Retrieval using {config['captions_type']} captions")
    
    start = time.time()
    opts = get_common_opts(params=params)
    opts.jatayu = os.path.isdir("/home/manugaur")
    
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

    """
    cocoid2idx gives UID. The idx is used to retrieve image features.
    text feats processed on the fly by the scorer.
    """
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

#------------------------------------------------------------------------------------------------------------------------------------------------    
    #load scorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config['scorer'] =="vitb32":
        model_name = "ViT-B/32"
        model, preprocess = clip.load(model_name, device=device)
    elif config['scorer'] =='siglip400m':
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
        preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    else:
        raise Exception("Choose a valid scorer")
    model.eval()

    if config['baseline'] is None:
        captioner = f"{config['data']}_{config['method']}"        
        preds_path = f"/home/manugaur/EGG/inference_preds/{captioner}.pkl"
        # preds_path = "/home/manugaur/nips_benchmark/baselines/instructblip2_vicuna7b_p2.pkl"

        with open(preds_path, "rb") as f:
            preds = pickle.load(f)
    else:
        preds = pickle.load(open(f"/home/manugaur/nips_benchmark/baselines/{config['baseline']}.pkl", 'rb'))
        if not isinstance(list(preds.values())[0], str):
            preds = {k: v[0] for k,v in preds.items()}
    
    all_bag_scores = []

    for bag_size in [3,5,7]: 
        bag_dir = "/home/manugaur/nips_benchmark/final_bags"
        
        #benchmark : list of bags. Each bag: list of cocoids    
        with open(os.path.join(bag_dir, f"{bag_size}.json"), "r") as f:
            listofbags = json.load(f)

        benchmark = []
        for bag in listofbags:
            benchmark.append([cocoid2idx[str(cocoid)] for cocoid in bag])
        
        for idx , bag in tqdm(enumerate(benchmark), total = len(benchmark)):
                

            bag_img_feats = img_feats[bag]
            batch_caps = [preds[int(idx2cocoid[_])].strip() for _ in bag]

            with torch.no_grad():
                if config['scorer'] == "siglip400m":
                    inputs = preprocess(text=batch_caps, return_tensors="pt", padding = True, truncation=True)
                    batch_text_feats = model.text_model(inputs["input_ids"].to(device))[1]

                else:
                    batch_text_feats = model.encode_text(clip.tokenize(batch_caps, context_length=77, truncate=True).to(device))
                

                bag_img_feats = bag_img_feats / bag_img_feats.norm(dim=-1, keepdim = True)
                batch_text_feats = batch_text_feats / batch_text_feats.norm(dim=-1, keepdim = True)
                

                _, acc = loss(batch_text_feats,bag_img_feats, training=False, get_acc_5 = False, aux_input=None)
                recall_1.append([_.item() for _ in acc['acc']])
                clip_s.append(acc['clip_s'].mean().item())         
                all_bag_scores.append([i.item() for i in acc['acc']])
        print(f"| BAG SIZE = {bag_size}")      

        with open(f"/home/manugaur/nips_benchmark/recall_per_bag/{config['baseline']}.json", "w") as f:
            json.dump(all_bag_scores, f)
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
    config['scorer'] = "vitb32" #[siglip400m, vitb32]
    config['baseline'] = None # baseline can be MLLM baselines or GT caps with multiple caps per image
    config['method'] =  "cider-sr_lambda_1e7_srlv_lr_1e7_g_baseline_curri" #"cider-sr_lambda_1_srlv_lr_1e7_g_baseline_curri"
    config['data'] = "blip2mistral"
    config["opts"]["batch_size"]= 100
    config['split'] = "test_val"
    config['use_greedy'] = False
    config['avg_text_feat'] = False 
    main(params, config)