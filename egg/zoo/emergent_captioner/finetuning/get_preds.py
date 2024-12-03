import sys
import time
import os
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import get_linear_schedule_with_warmup
from egg.zoo.emergent_captioner.finetuning.utils import get_config, process_config, get_cl_args, init_wandb, get_best_state_dict, int2mil
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
os.environ["WANDB_API_KEY"] = "b389b1a0f740ce1efcfd09b332fd3a83ef6130fe"
    

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
        batch_size=100,
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

    #test
    data_kwargs["batch_size"] = config["inference"]["batch_size"]
    data_kwargs["mle_train"] = False
    test_loader = wrapper.get_split(split="test", caps_per_img = 5, neg_mining = False, **data_kwargs)

    game = build_game(opts, config)
    # print_grad_info(game)

    # if config['freeze_wte']:
    #     for p in game.sender.clipcap.gpt.lm_head.parameters():
    #         p.requires_grad = False
    #     for p in game.sender.clipcap.gpt.transformer.wte.parameters():
    #         p.requires_grad = False
    
    # if config["diff_lr"]:
    #     optimizer = torch.optim.AdamW(
    #         [
    #             {"params": game.receiver.clip.parameters()},
    #             {"params": game.sender.clipcap.clip_project.parameters(), "lr" : 5e-9},
    #             {"params": game.sender.clipcap.gpt.parameters()},
    #         ],
    #         lr=opts.lr,
    #     )
    # else:
    optimizer = torch.optim.AdamW(game.sender.parameters(), lr = opts.lr)
    # optimizer = torch.optim.Adam(game.sender.parameters(), lr=opts.lr)

    # Create trainers object
    if config["train_method"] == "mle":
        total_steps = opts.n_epochs* len(train_loader)
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
                ModelSaver(opts),
            ],
            debug=opts.debug,
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
            ModelSaver(opts),
        ],
        debug=opts.debug,
        )  
    
    if opts.distributed_context.is_distributed:
        trainer.game = trainer.game.module

    if config['mllm'] == "clipcap" : #and config["train_method"] != "mle":   
        trainer.game.sender.patch_model(batch_size = opts.batch_size, prefix_len = config['prefix_len'], )

    # #patching unfreezes wte. If finetuning CLIP with fully frozen GPT, run this :     
    # if config['freeze_wte']:
    #     for p in trainer.game.sender.clipcap.gpt.lm_head.parameters():
    #         p.requires_grad = False
    #     for p in game.sender.clipcap.gpt.transformer.wte.parameters():
    #         p.requires_grad = False

    #Training
    if not config["ONLY_INFERENCE"]:
        trainer.train(config, opts)

    #Get inference preds
    if not os.path.isdir(config["inference"]["output_dir"]):
        os.makedirs(config["inference"]["output_dir"])

    # getting MLE preds : comment this and path to inference_preds and inference_log

    if config['mllm'] == "clipcap":   
        trainer.game.sender.unpatch_model()
        trainer.game.sender.clipcap.load_state_dict(get_best_state_dict(config))
        trainer.game.sender.patch_model(batch_size = config["inference"]["batch_size"], prefix_len = config['prefix_len'], )
        
        trainer.game.sender.clipcap.load_state_dict(get_best_state_dict(config))

        config["WANDB"]["logging"] = False

        if config["train_method"] != "mle":
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

    #hparams
    config["inference"]["batch_size"] = 100


    params = get_cl_args(config)


    main(params, config)