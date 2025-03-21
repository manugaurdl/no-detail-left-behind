import sys
import time
import os
import torch
import numpy as np
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

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def get_loader(wrapper, bag_size, data_kwargs):
    return wrapper.get_split(split="train", caps_per_img= config["CAPS_PER_IMG_train"], bag_size = bag_size,  **data_kwargs)
    

def main(params, config):
    start = time.time()
    opts = get_common_opts(params=params)
    opts.fp16 = config['fp16']
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
    wrapper = name2wrapper[opts.train_dataset](captions_type = config["captions_type"], dataset_dir = opts.dataset_dir  , curri = config["curri"])
    
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
    
    #data loaders
    train_loaders = {bag_size : get_loader(wrapper, bag_size, data_kwargs) for bag_size in config['curri'].values()}

    data_kwargs["batch_size"] = config["inference"]["batch_size"]
    data_kwargs["mle_train"] = False
    val_loader = wrapper.get_split(split="val", caps_per_img = config["CAPS_PER_IMG_val"], bag_size = 0, **data_kwargs)
    test_loader = wrapper.get_split(split="test", caps_per_img = config["CAPS_PER_IMG_val"], bag_size = 0, **data_kwargs)
    
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

    #if we are training MLE
    if config["train_method"] == "mle" and not config['ONLY_INFERENCE']:
        n_epochs = list(config['curri'].keys())[-1]
        total_steps = n_epochs* len(train_loaders[0])
        scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(total_steps * config["warmup_ratio"]), num_training_steps= total_steps)

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            optimizer_scheduler = scheduler,
            train_loaders = train_loaders,
            val_loader= val_loader,
            test_loader = test_loader,
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
        val_loader= val_loader,
        test_loader = test_loader,
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
    
    load_best_model(trainer, config)
    trainer.game.sender.patch_model(batch_size = opts.batch_size, prefix_len = config['prefix_len'])

    config["WANDB"]["logging"] = False
    trainer.train(config, opts, inference = True) #init_val is run. val_data = inference data if inference = True.

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


    main(params, config)