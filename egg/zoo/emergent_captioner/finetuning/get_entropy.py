import sys
import time
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import random
from transformers import get_linear_schedule_with_warmup
from egg.zoo.emergent_captioner.finetuning.utils import get_config, process_config, get_cl_args, init_wandb, get_best_state_dict, int2mil, load_prev_state, load_best_model
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
from egg.zoo.emergent_captioner.evaluation.entropy import calc_entropy

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"
# os.environ["CUDA_VISIBLE_DEVICES"] = str((0))
os.environ["WANDB_API_KEY"] = "b389b1a0f740ce1efcfd09b332fd3a83ef6130fe"

def get_loader(wrapper, level_bsz, data_kwargs):
        level, bsz = level_bsz.split("_")
        print(level_bsz)
        if level == "rand":
            return wrapper.get_split(split="train", caps_per_img= config["CAPS_PER_IMG_train"], neg_mining = False,  **data_kwargs)
        else:
            return wrapper.get_split(split="train", caps_per_img= config["CAPS_PER_IMG_train"], neg_mining = True, level_bsz = level_bsz,  **data_kwargs)
    

def main(params, config):
    start = time.time()
    opts = get_common_opts(params=params)
    opts.fp16 = config['fp16']
    opts.jatayu = os.path.isdir("/home/manugaur")
    opts.loss_type= config['train_method']
    store_job_and_task_id(opts)
    setup_for_distributed(opts.distributed_context.is_leader)
    if opts.distributed_context.local_rank ==0:
        init_wandb(config)

    print(get_sha())
    
    name2wrapper = {
        "conceptual": ConceptualCaptionsWrapper,
        "coco": CocoWrapper,
        "flickr": FlickrWrapper,
    }
    # args
    wrapper = name2wrapper[opts.train_dataset](captions_type = config["captions_type"], dataset_dir = opts.dataset_dir, jatayu = opts.jatayu, neg_mining = config["neg_mining"])
    
    data_kwargs = dict(
        batch_size=opts.batch_size,
        transform=get_transform(opts.sender_image_size, opts.recv_image_size),
        mllm = config['mllm'],
        num_workers=config["num_workers"],
        seed=opts.random_seed,
        debug = config['DEBUG'],
        mle_train = config["train_method"] =="mle",
        max_len_token = opts.max_len,
        prefix_len = config["prefix_len"],
        is_dist_leader = opts.distributed_context.is_leader,
    )
    
    #train
    train_loaders = {f"{i[0]}_{i[1]}" : get_loader(wrapper, f"{i[0]}_{i[1]}", data_kwargs) for i in config["neg_mining"]["curricullum"].values()}
    #val
    if config['train_method'] == "mle":
        val_loader_rand = wrapper.get_split(split="val", caps_per_img = 5, neg_mining = False,  **data_kwargs)
        val_loader_neg = None
    else:
        if config['neg_mining']['val_level'] =="rand":
            val_loader_rand = wrapper.get_split(split="val", caps_per_img = config["CAPS_PER_IMG_val"], neg_mining = False,  **data_kwargs)
            val_loader_neg = None
            
        else:
            val_loader_neg = wrapper.get_split(split="val", caps_per_img = config["CAPS_PER_IMG_val"], neg_mining = True, level = config['neg_mining']['val_level'],  **data_kwargs)
            val_loader_rand = None

    #test
    data_kwargs["batch_size"] = config["inference"]["batch_size"]
    data_kwargs["mle_train"] = False
    test_loader = wrapper.get_split(split="test", caps_per_img = config["CAPS_PER_IMG_val"], neg_mining = False, **data_kwargs)
    # for idx, batch in tqdm(enumerate(train_loader),total = len(train_loader)):
    #     pass

    game = build_game(opts, config)
    # print_grad_info(game)


    if config["finetune_model"]=="clip":
        optimizer = torch.optim.AdamW(
            [
                {"params": game.sender.clip.visual.parameters()},
                {"params": game.sender.clipcap.clip_project.parameters()},
            ],
            lr=opts.lr,
        )
    else:
        optimizer = torch.optim.AdamW(game.sender.parameters(), lr = opts.lr)
        # optimizer = torch.optim.Adam(game.sender.parameters(), lr=opts.lr)
    
    if config['resume_training']['do']:
        optim_state_dict, game_state_dict = load_prev_state(config, game)
        optimizer.load_state_dict(optim_state_dict)
        game.sender.load_state_dict(game_state_dict)
        
    # Create trainers object
    if config["train_method"] == "mle" and not config['ONLY_INFERENCE']:
        n_epochs = [_ for _ in config['neg_mining']['curricullum'].keys()][-1]
        total_steps = n_epochs* len(train_loaders['rand'])
        scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(total_steps * config["warmup_ratio"]), num_training_steps= total_steps)

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_loaders = train_loaders,
            optimizer_scheduler = scheduler,
            validation_data_rand =val_loader_rand,
            validation_data_neg =val_loader_neg,
            inference_data = test_loader,
            callbacks=[
                ConsoleLogger(as_json=True, print_train_loss=True),
                ModelSaver(opts, config),
            ],
            debug=opts.debug,
            config= config,
        )
    else:
        trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_loaders = train_loaders,
        validation_data_rand =val_loader_rand,
        validation_data_neg =val_loader_neg,
        inference_data = test_loader,
        callbacks=[
            ConsoleLogger(as_json=True, print_train_loss=True),
            ModelSaver(opts, config),
        ],
        debug=opts.debug,
        config= config,
        )  
    
    if opts.distributed_context.is_distributed:
        trainer.game = trainer.game.module

    if config['mllm'] == "clipcap" : #and config["train_method"] != "mle":   
        trainer.game.sender.patch_model(batch_size = opts.batch_size, prefix_len = config['prefix_len'], )

    #patching unfreezes wte. If finetuning CLIP with fully frozen GPT, run this :     
    if config['freeze_wte']:
        for p in trainer.game.sender.clipcap.gpt.lm_head.parameters():
            p.requires_grad = False
        for p in game.sender.clipcap.gpt.transformer.wte.parameters():
            p.requires_grad = False

    #Training
    if not config["ONLY_INFERENCE"]:
        trainer.train(config, opts)

    #Get inference preds
    if not os.path.isdir(config["inference"]["output_dir"]):
        os.makedirs(config["inference"]["output_dir"])

    # # getting MLE preds : comment this and path to inference_preds and inference_log


    # filename = "mistral_mle_final"
    load_best_model(trainer, config)
    filename = '_'.join(config['opts']['checkpoint_dir'].split('/')[-2:])

    print(filename)
    preds = pickle.load(open(f"/home/manugaur/EGG/inference_preds/{filename}.pkl", 'rb'))

    id2entropy = calc_entropy(trainer.game.sender, preds)
    print(f'entropy --> {np.array(list(id2entropy.values())).mean()}')
        
    with open(f"/home/manugaur/EGG/entropy/{filename}", "wb") as f:
        pickle.dump(id2entropy, f)
    exit()


    
    
    config["WANDB"]["logging"] = False

    trainer.train(config, opts, inference = True) #init_val is run. val_data = inference data if inference = True.

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # torch.set_deterministic(True)
    use_ddp = False    

    if "LOCAL_RANK" in os.environ:
        use_ddp = True
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config_filename = f"egg/zoo/emergent_captioner/finetuning/configs/{sys.argv[1:][0]}.yml"
    config = get_config(config_filename)
    config = process_config(config, use_ddp, sys.argv[1:])
    params = get_cl_args(config)


    main(params, config)