True# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, NamedTuple, Optional
import os
import yaml
import torch
from transformers import GPT2LMHeadModel, LogitsProcessor, get_linear_schedule_with_warmup
from egg.core import Callback, Interaction
import wandb

class MyCheckpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]
    opts: argparse.ArgumentParser

class KLRegularizer:
    def __init__(self, device=torch.device("cuda")):
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    @torch.no_grad()
    def compute_kl_loss(self, indices, log_probs):
        # 50256 is gpt2 beginning of sentence
        indices = torch.cat([torch.ones_like(indices[:, :1]) * 50256, indices], dim=1)
        # we take probs from bos until last token
        # print(f"------->{self.lm.device}")
        # for i in self.lm.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")
        generated = self.lm(indices)["logits"].log_softmax(-1)[:, :-1, :]

        step_kl_div = []
        for timestep in range(generated.shape[1]):
            x = torch.nn.functional.kl_div(
                log_probs[:, timestep],
                generated[:, timestep],
                log_target=True,
                reduction="none",
            )
            step_kl_div.append(x.sum(-1))  # summing over vocab_dim
        kl_div = torch.stack(step_kl_div, dim=1)
        return kl_div


class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, do_sample):
        self.eos_token_id = tokenizer.eos_token_id #50256

        # f : tokenizer.convert_ids_to_tokens --> decodes token  i.e f(500) = "walk" 
        # There are 121 strings that contain "." --> all of them are treated as stop tokens
        self.stop_word_ids = set(
            [
                idx
                for idx in range(len(tokenizer))
                if "." in tokenizer.convert_ids_to_tokens(idx)
            ]
        )
        self.vocab_size = len(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # iterate each batch of prefix tokens ; input_ids (B , 10)
        for i, input_id in enumerate(input_ids):
            if input_id[-1].item() in self.stop_word_ids:
                scores[i, : self.vocab_size] = torch.finfo().min #-1e+4 # #-65504
                scores[i, self.vocab_size :] = float("-inf") #-1e+4 #  float("-inf") #-1e+4
                scores[i, self.eos_token_id] = 0.0
        return scores


class ModelSaver(Callback):
    def __init__(self, opts: argparse.ArgumentParser, config : dict):
        self.opts = opts
        self.config = config
    def get_checkpoint(self):
        self.is_ddp = self.trainer.distributed_context.is_distributed

        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )

        # if self.is_ddp:
        #     game = self.trainer.game.module
        #     self.trainer.game.module.loss.remove_fields_negatives()
        #     self.trainer.game.module.sender.unpatch_model()

        # else:
        game = self.trainer.game
        # cleaning a model such that it has default settings e.g. no buffer and no modules/tensors in the loss
        # this is done to avoid mandatory fields when loading a model e.g. a tensor of negatives
        self.trainer.game.loss.remove_fields_negatives()

        self.trainer.game.sender.unpatch_model()
        
        return MyCheckpoint(
            epoch=self.epoch,
            model_state_dict=game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            optimizer_scheduler_state_dict=optimizer_schedule_state_dict,
            opts=self.opts,
        )

    def save_clipclap_model(self, epoch=None, model_name = None, SAVE_BEST_METRIC = None, save_epoch = False):
        self.is_ddp = self.trainer.distributed_context.is_distributed

        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path
                and self.trainer.distributed_context.is_leader
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)

                self.trainer.game.sender.unpatch_model()
                
                model_name  = f"e_{epoch if epoch else 'final'}.pt"
                
                param_keys = [n for n,p in self.trainer.game.sender.named_parameters() if p.requires_grad]
                state_dict = {k: v for k, v in self.trainer.game.sender.state_dict().items() if k in param_keys}

                

                """following block also preserves only trainable params"""
                # for name in list(x.keys()):
                #     condition = 'lora' in name  or 'clip_project' in name

                #     if condition:
                #         continue
                #     else:
                #         x.pop(name)
                
                if SAVE_BEST_METRIC:
                    torch.save(
                        state_dict,
                        self.trainer.checkpoint_path / "best.pt",
                    )
                if save_epoch:
                    torch.save(x, str(self.trainer.checkpoint_path / model_name))


                # optimizer path corresponds to the best.pt checkpoint. Want optim state dict for 7th epoch ? --> train only till 7 epochs!
                
                if save_epoch:
                    optimizer_path =  os.path.join(str(self.trainer.checkpoint_path) ,f"optimizer{epoch}.pth")
                    torch.save(self.trainer.optimizer.state_dict(), optimizer_path)
                # if self.is_ddp:
                #     self.trainer.game.module.sender.patch_model()
                # else:         
                
                self.trainer.game.sender.patch_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int, model_name : str, SAVE_BEST_METRIC: bool, save_epoch : bool):
        self.epoch = epoch
        if self.opts.captioner_model == "clipcap":
            self.save_clipclap_model(epoch=epoch, model_name = model_name, SAVE_BEST_METRIC = SAVE_BEST_METRIC, save_epoch = save_epoch)

    def on_train_end(self, epoch : int, model_name : str):

        try:
            isinstance(self.epoch, ModelSaver)
        except:
            self.epoch = epoch
        
        if self.opts.captioner_model == "clipcap":
            self.save_clipclap_model(model_name = model_name)

def get_config(filename):
    config_path = os.path.join(os.getcwd(),filename)
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

def process_config(config, use_ddp, sys_args):
    """
    /home/manugaur —> /ssd_scratch/cvit/manu
    """
    if use_ddp:
        config["num_workers"] = 0

    config["captions_type"] = sys_args[1]
    config["opts"]["checkpoint_dir"] = os.path.join(config['opts']['checkpoint_dir'].split("checkpoints")[0], f"checkpoints/{sys_args[1] + '/' + sys_args[0].split('_')[0]}_{config['WANDB']['run_name']}") #data/method
    config["WANDB"]["run_name"] = f"{sys_args[0].split('_')[0]}_{sys_args[1]}_{config['WANDB']['run_name']}"#{method}_{data}
    # if "mle_model_path" in config["opts"]:
    #     config["opts"]["mle_model_path"] = os.path.join(config['opts']['mle_model_path'].split("/checkpoints")[0], f"checkpoints/{sys_args[1]}/mle_final/best.pt") #mle_1_train_cap    
        # print(f"| Loaded MLE model :{config['opts']['mle_model_path']}")

    if config["ONLY_INFERENCE"] or config["ONLY_VAL"]:
        config["WANDB"]["logging"] = False
    
    if config["DEBUG"]:
        # config["SAVE_BEST_METRIC"] = False
        config["opts"]["checkpoint_freq"] = 0
        config["WANDB"]["logging"] = False
    return config

def get_cl_args(config):
    params = []
    for k,v in config['opts'].items():
        params.append(f"--{k}")
        params.append(f"{v}")
    return params

def init_wandb(config):
    if config['WANDB']['logging'] and (not config['WANDB']['sweep']) :
        wandb.init(entity= config["WANDB"]["entity"], project=config["WANDB"]['project'], config = config)
        wandb.run.name = config['WANDB']['run_name']

def get_best_state_dict(config):
    print("| LOADED BEST MODEL FOR INFERENCE")
    
    desired_format_state_dict = torch.load(config["official_clipcap_weights"])
    if config["SAVE_BEST_METRIC"] or config['ONLY_INFERENCE']:
        print(f"checkpoint loaded : {os.path.join(config['opts']['checkpoint_dir'], 'best.pt')}")
        saved_state_dict = torch.load(os.path.join(config["opts"]["checkpoint_dir"], "best.pt"))#[1]
    else:
        saved_state_dict = torch.load(os.path.join(config["opts"]["checkpoint_dir"], "e_10.pt"))#[1]

    return saved_state_dict

def int2mil(number):
    if abs(number) >= 100_000:
        formatted_number = "{:.1f}M".format(number / 1_000_000)
    else:
        formatted_number = str(number)
    return formatted_number

def trainable_params(model):
    # print(f'{int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))} trainable params')
    return int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))
    # return sum(p.numel() for p in model.parameters() if p.requires_grad == True)

def load_prev_state(config, game):
        load_dir = os.path.join(config['opts']['checkpoint_dir'].split(config['captions_type'])[0], config['captions_type'], config['resume_training']['dir'])
        
        optim_state_dict = torch.load(os.path.join(load_dir, f"optimizer{config['resume_training']['load_epoch']}.pth"))

        trained_wts = torch.load(os.path.join(load_dir, f"e_{config['resume_training']['load_epoch']}.pt")) #lora params and clip_project
        updated_weights = game.sender.state_dict().copy()

        for k in list(game.sender.state_dict().keys()):
            if k in trained_wts:
                updated_weights[k] = trained_wts[k]
        
        return optim_state_dict, updated_weights

def load_best_model(trainer, config):

    if config['mllm'] == "clipcap":   
        trainer.game.sender.unpatch_model()

    trained_wts = torch.load(os.path.join(config['opts']['checkpoint_dir'], "best.pt"))
    model_params = trainer.game.sender.state_dict().copy()
    for k in model_params.keys():
        if k in trained_wts:
            model_params[k] = trained_wts[k]
    
    trainer.game.sender.load_state_dict(model_params)
    

    # for k in list(trainer.game.sender.state_dict().keys()):
    #     if 'sender.' + k in trained_wts:
    #         updated_wts[k] = trained_wts['sender.' + k]
    #     elif k in trained_wts:
    #         updated_wts[k] = trained_wts[k]

    # trainer.game.sender.load_state_dict(updated_wts)
    # trainer.game.sender.patch_model(batch_size = config["inference"]["batch_size"], prefix_len = config['prefix_len'])

    print(f"| LOADED BEST MODEL FOR INFERENCE ON TEST+VAL SET : {os.path.join(config['opts']['checkpoint_dir'], 'best.pt')}")