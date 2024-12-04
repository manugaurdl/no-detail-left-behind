# Global vars
import time
import os
import wandb
import pathlib
import pickle
import json
import numpy as np
from typing import List, Optional
from prettytable import PrettyTable
from tqdm import tqdm
import torch.distributed as dist
from collections import defaultdict
try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .batch import Batch
from .callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
)
from .distributed import get_preemptive_checkpoint_dir
from .interaction import Interaction
from .util import get_opts, move_to
from egg.zoo.emergent_captioner.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    print_grad_info,
    setup_for_distributed,
    store_job_and_task_id,
)
from egg.zoo.emergent_captioner.evaluation.evaluate_nlg import compute_nlg_metrics
from egg.zoo.emergent_captioner.finetuning.losses import DiscriminativeLoss
from egg.zoo.emergent_captioner.evaluation.mmvp_mllm import mmvp_mllm_benchmark
from egg.zoo.emergent_captioner.evaluation.mmvp_vlm import mmvp_vlm_benchmark
# from egg.zoo.emergent_captioner.evaluation.winoground_vlm import winoground_vlm
from egg.zoo.emergent_captioner.evaluation.bag_eval import eval_on_bags

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass

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

def get_loader(epoch, ranges):
    prev_end = 0
    sorted_items = sorted(ranges.items(), key=lambda x: x[1])

    for key, end in sorted_items:
        if prev_end <= epoch <end:
            return key
        prev_end = end
    raise Exception("epoch out of curricullum")

def get_ds(epoch, config):
    for start_epoch, level_bsz in config['neg_mining']['curricullum'].items():
        level, bsz = level_bsz
        if start_epoch > epoch:
            return f"{level}_{bsz}"

def count_trainable_parameters(model):
    table = PrettyTable(["Modules", "Requires grad", "Trainable parameters"])
    table.align["Modules"] = "l"
    table.align["Requires grad"] = "c"
    table.align["Trainable parameters"] = "r"

    total_params = 0
    for name, parameter in model.named_parameters():
        
        requires_grad = True
        params = parameter.numel()
        
        if not parameter.requires_grad:
            requires_grad = False
            params = 0
        
        table.add_row([name, requires_grad, params])
        total_params += params

    print(table)
    print(f"Total Trainable Params: {total_params:,}")
    return total_params

base_dir = "/home/manugaur"
a100_dir = "/home/ubuntu/pranav/pick_edit"
if os.path.isdir(a100_dir):
    base_dir = a100_dir


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loaders: dict,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data_rand: Optional[DataLoader] = None,
        validation_data_neg: Optional[DataLoader] = None,
        inference_data : Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
        debug: bool = False,
        config:dict = None,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_loaders =  train_loaders
        self.val_loader_rand = validation_data_rand
        self.val_loader_neg = validation_data_neg
        self.inference_loader = inference_data 
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.debug = debug

        self.should_stop = False
        
        self.start_epoch = 0
        if config['resume_training']['do']:
            self.start_epoch = config['resume_training']['load_epoch'] # start epoch should be the next one but ckpts are indexed from 1.
        
        
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        self.update_freq = common_opts.update_freq
        self.STEP = 0
        if common_opts.load_from_checkpoint is not None:
            print(
                f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}"
            )
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print("# Distributed context: ", self.distributed_context)

        if self.distributed_context.is_leader and not any(
            isinstance(x, CheckpointSaver) for x in self.callbacks
        ):
            if common_opts.preemptable:
                assert (
                    common_opts.checkpoint_dir
                ), "checkpointing directory has to be specified"
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = (
                    None
                    if common_opts.checkpoint_dir is None
                    else pathlib.Path(common_opts.checkpoint_dir)
                )

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_freq=common_opts.checkpoint_freq,
                )
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert (
                common_opts.tensorboard_dir
            ), "tensorboard directory has to be specified"
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        if self.distributed_context.is_distributed:
            print(f"Wrapping model to GPU:{self.distributed_context.local_rank}")
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            self.game.to(device_id)

            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper.
            #    This wrapper would sync gradients of the underlying tensors - which are the ones that optimizer
            #    holds itself.  As a result it seems to work, but only because DDP doesn't take any tensor ownership.

            self.game = torch.nn.parallel.DistributedDataParallel(
                self.game,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
            )
            self.optimizer.state = move_to(self.optimizer.state, device_id)

        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)

        if common_opts.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def prepare_logs(self,summary, loss, interaction, reward, train_method, loss_type, epoch, config):

        val_log = { "Val Loss" : loss,
                    "Val Reward" : reward,
                    "CIDEr" : summary["CIDEr"],
                    "Bleu_4" : summary["Bleu_4"],
                    'METEOR': summary["METEOR"],
                    "ROUGE_L" : summary['ROUGE_L'],
                    }

        if config['log_spice'] and config['captions_type']!="blip2mistral":
            val_log["SPICE"] = summary["SPICE"]

        if config["finetune_model"] == "clip":
            val_log["mmvp_avg"] = interaction['mmvp_avg']
            # val_log["recall_5_clip_zs"] = interaction.aux['recall_5_clip_zs'].mean()
            # val_log["recall_1_clip_zs"] = interaction.aux["recall_1_clip_zs"].mean()
            # val_log.update(interaction['wino'])

            if config["WANDB"]["log_mmvp_all"]:
                val_log.update(interaction['mmvp_all'])
        # if WANDB.log _mmvp_aspects:
        #     val_log.update(all mmvp aspects from interaction.aux)

        if train_method == "mle":
            del val_log["Val Loss"]
            del val_log["Val Reward"]

        """
        metric decides how you save model. If clip_ft : save model with highest mmvp.
        """
        if loss_type == 'discriminative' or config["CIDER_SR"]:
            # if config['finetune_model'] == "llm":
            metric =  interaction['acc'].mean().item()
            # elif config['finetune_model']=="clip":
            #     # metric = interaction['mmvp_avg']
            #     metric = interaction['wino']['wino_text_rand_25']
            val_log["VAL_R@1"] = interaction['acc'].mean().item()
        
        else:
            metric = summary["CIDEr"]            

        # aggregated print for 1st obj, pass for other 2
        for callback in self.callbacks:
            callback.on_validation_end(loss, interaction, epoch + 1)

        return val_log, metric

    def run_validation(self, loader, epoch : int, config : dict,  inference : bool = False):
        validation_loss = validation_interaction = None
        if (
            loader is not None
            and self.validation_freq > 0
            and (epoch + 1) % self.validation_freq == 0
        ):
            # for idx, callback in enumerate(self.callbacks): 
            #     if idx in [0,1]:
            #         continue
            #     callback.on_validation_begin(epoch + 1) # pass
            validation_loss, validation_interaction, val_reward, summary = self.eval(loader, inference = inference, config = config, GREEDY_BASELINE = self.GREEDY_BASELINE, train_method = self.train_method)

            val_log, metric = self.prepare_logs(summary, validation_loss, validation_interaction, val_reward, self.train_method, self.opts.loss_type, epoch, config)

            return val_log, validation_interaction, metric

    def log(self, log, interaction, metric, epoch, name, config = None, inference=False):
            """val log is plotted on wandb. Inference log is saved as json."""

            if inference:
                #save inference preds
                self.save_val_preds(interaction, config, inference = True)
                
                # save inference log
                # test_log = {}
                # test_log['recall_1'] = interaction['acc'].mean().item()
                # test_log['recall_5'] = interaction['acc_5'].mean().item()
                # test_log['CLIP_s'] = interaction['clip_s'].mean().item()
                # test_log.update(log)

    
                # with open(os.path.join(inference_log_dir,  f"{config['captions_type']}_{config['opts']['checkpoint_dir'].split('/')[-1]}.json"), "w") as f:
                #     json.dump(test_log, f)    
                # with open("/home/manugaur/EGG/inference_log/blip2mistral_mle.json", "w") as f:
                #     json.dump(test_log, f)

            else:

                log["epoch"] = epoch
                log["val_log_prob"] =  np.array(interaction['log_prob']).mean()

                if name == "rand":
                    wandb.log(log, step = self.STEP)
                else:
                    wandb.log({"VAL_R@1_NEG" : log["VAL_R@1"]}, step = self.STEP)



    def rand_neg_val(self, epoch : int, WANDB : bool,  config : dict, inference : bool = False):
        
        if inference:
            test_log, interaction, metric = self.run_validation(self.inference_loader, epoch, config, inference)
            self.log(test_log, interaction, metric, None, None, config = config, inference = inference)
        
        else:
            if self.val_loader_rand is not None:
                rand_log, rand_interaction, metric = self.run_validation(self.val_loader_rand, epoch, config, inference)
            if self.val_loader_neg is not None:
                rand_log, rand_interaction, metric = self.run_validation(self.val_loader_neg, epoch, config, inference)

            if WANDB:
                self.log(rand_log, rand_interaction, metric, epoch, "rand", config = config)
                
                if self.val_loader_neg is not None:
                    self.log(rand_log, rand_interaction, metric, epoch, "neg", config = config)
            
            if self.SAVE_USING_NEG and self.val_loader_neg is not None:
                return metric

        torch.cuda.empty_cache()
        return metric

    def eval(self, loader, inference : bool, config : dict, data=None, GREEDY_BASELINE = False, train_method = None):
        
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        self.game.eval()

        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(loader), total = len(loader)):
                if 522198 not in batch[1] and 190756 not in batch[1] and 279806 not in batch[1]:
                    continue
                print("Generating")
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                
                if config['dataset'] =="imagecode":
                    interactions.append(self.game(*batch, train_method = train_method, inference=inference))
                    continue
                else:    
                    optimized_loss, interaction, reward = self.game(*batch, train_method = train_method, GREEDY_BASELINE= config['GREEDY_BASELINE'], inference=inference, CIDER_SR = config['CIDER_SR'])
                
                
                """
                interaction : sender_input=None, receiver_input=None, labels = tensor, aux_input = {cocoid, captions, tokens, mask}, message, receiver_output=None, message_length=None, aux = {"kl_div" = torch.rand(1)}
                """
                # lst = []
                # for _ in range(self.distributed_context.world_size):
                #     lst.append(torch.zeros_like(interaction.labels))

                # dist.all_gather(lst, interaction.labels)
                # interaction = torch.cat(lst, dim=0).to("cpu")
                # torch.save(interaction, "/ssd_scratch/cvit/manu/temp_interaction.pt")

                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                for callback in self.callbacks:
                    callback.on_batch_end(
                        interaction, optimized_loss, n_batches, is_training=False
                    )
                if interaction.sender_input is None:
                    interaction.sender_input = torch.rand((3,1)).float()
                
                if inference or isinstance(self.game.loss, DiscriminativeLoss):
                    if interaction.receiver_input is None:
                        interaction.receiver_input = torch.rand((3,1)).float()
                    #only for SR loss --> during SR training or MLE/CiDEr inference
                    interaction.aux["mean_rank"] = interaction.aux["mean_rank"].view(1,1)
                    interaction.aux["median_rank"] = interaction.aux["median_rank"].view(1,1) 
            
                interactions.append(interaction)
                n_batches += 1
                torch.cuda.empty_cache()
        if config['dataset'] =="imagecode":
            acc = np.array([x['acc'].item() for x in interactions]).mean()
            print(f"IMAGE CoDe R@1 : {acc} ")
            exit()

        mean_loss /= n_batches
        #if data is dict/tensor --> its gets extended N_batch times. If its a list, a new list of list gets created of len = N_batch
        # full_interaction = Interaction.from_iterable(interactions)
        mean_loss /= n_batches
        #if data is dict/tensor --> its gets extended N_batch times. If its a list, a new list of list gets created of len = N_batch
        # full_interaction = Interaction.from_iterable(interactions)
        
        full_interaction  = defaultdict(list)
        if config['train_method']=="discriminative" or config['CIDER_SR']:
            for interaction in interactions:
                for k,v in interaction.aux.items():
                    full_interaction[k].append(v.item())
            full_interaction  = {k: np.mean(v).mean() for k,v in full_interaction.items()}

        if config['dataset'] =="imagecode":
            print("| IMAGE CODE results : ")
            for k,v in full_interaction.items():
                print(f"{k} = {v:.3f}")
            exit()
        
        full_interaction['cocoid'] = []
        for interaction in interactions:
            for _ in interaction.aux_input['cocoid']:
                full_interaction['cocoid'].append(_.item())

        full_interaction['message'] = [interaction.message for interaction in interactions]

        full_interaction['captions'] = [interaction.aux_input['captions'] for interaction in interactions]
        
        img_ids = full_interaction['cocoid']

        # #TMLR: beam search
        # caps = full_interaction['message'][0]
        # cocoids = [522198,190756,279806]
        # indices = [img_ids.index(i) for i in cocoids]
        # for _, idx in enumerate(indices):
        #     print(cocoids[_])
        #     print(caps[idx])
        # exit()

        captions = full_interaction['captions']
        preds_per_batch = full_interaction['message']
        bsz = len(preds_per_batch[0])
        gold_standard = {}
        
        for i, batch in enumerate(captions):
            coco_caps = list(zip(*batch))
            for j, img in enumerate(coco_caps):
                gold_standard[(i)*bsz + j] = [{"caption": cap} for cap in img]
        
        predictions = {}
        
        for i, batch in enumerate(preds_per_batch):
            for j, pred in enumerate(batch):
                predictions[(i*bsz) + j] = [{"caption" :pred}]
        
        summary = compute_nlg_metrics(predictions, gold_standard, spice = config["log_spice"] and config['captions_type']!="blip2mistral") # score for each idx stored in summary except bleu


        # MMVP eval
        if config['finetune_model'] == "clip":

            mmvp_results = mmvp_vlm_benchmark(self.game.sender.clip, self.game.sender.clip_preproc, os.path.join(base_dir, "mmvp_vlm"))
            full_interaction["mmvp_avg"] =  np.array(list(mmvp_results.values())).mean()
            full_interaction.update({"mmvp_all" : mmvp_results})

            # wino_scores = winoground_vlm(self.game.sender.clip, self.game.sender.clip_preproc)
            # full_interaction["wino"] = wino_scores
        return mean_loss.item(), full_interaction, reward, summary

    def train_epoch(self,loader, WANDB, GREEDY_BASELINE, train_method, opts, config, epoch):

        mean_loss = 0
        n_batches = 0
        interactions = []
        
        self.game.train()
        if config['mllm']=="clipcap":
            if config["finetune_model"]=="llm":
                self.game.sender.clip.eval()
            else:
                self.game.sender.clip.train()
                self.game.sender.clipcap.gpt.eval()
        
        elif config['mllm']=="llava-phi":
            #temp : to make llava phi work
            self.game.sender.model.vision_tower.eval()
            # self.game.sender.model.language_model.eval()
            ########
        elif config['mllm']=="llava":
            pass

        self.game.receiver.eval()
        self.optimizer.zero_grad()

        for batch_id, batch in tqdm(enumerate(loader), total = len(loader)):
            # batch.append(GREEDY_BASELINE)
            if self.debug and batch_id == 10:
                break
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

            # context = autocast() if self.scaler else nullcontext()
            # with context:
            optimized_loss, interaction, reward = self.game(*batch, GREEDY_BASELINE, train_method, contrastive = config['contrastive'], reinforce = config["reinforce"], CIDER_SR = config['CIDER_SR'])
            #not accumulating gradients currently
            if self.update_freq > 1:
                # throughout EGG, we minimize _mean_ loss, not sum
                # hence, we need to account for that when aggregating grads
                optimized_loss = optimized_loss / self.update_freq


            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()
                # nn.utils.clip_grad_norm_(self.game.sender.model.multi_modal_projector.parameters(), 1.0)
                # print(f'----> {torch.norm(self.game.sender.model.language_model.lm_head.weight.grad)}')
                # print(f'----> {torch.norm(self.game.sender.model.multi_modal_projector.linear_1.weight.grad)}')
                # print(f'----> {self.game.sender.model.multi_modal_projector.linear_1.weight.grad}')
                # print(f'----> {self.game.sender.model.multi_modal_projector.linear_2.weight.grad}')

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            # print("before gather")
            # print(interaction.aux["acc"])

            # if (self.distributed_context.is_distributed and self.aggregate_interaction_logs):
            #     interaction = Interaction.gather_distributed_interactions(interaction)
            
            # print("after gather")
            # print(interaction.aux["acc"])

            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)
            print(f"Loss : {optimized_loss.item():.5f}")
            print(f"Avg Loss : {(mean_loss.item())/n_batches:.5f}")
            

            train_log = {"Loss" :optimized_loss.item(),
                        "lr" : self.optimizer.state_dict()["param_groups"][0]["lr"]
                         }
            if config['contrastive']:
                train_log["contrastive"] = interaction.aux['contrastive'].item()

            if self.opts.loss_type != 'cider' and not config['reinforce']:
                train_log["acc@1"] : interaction.aux['acc'].mean().item()
            
            if config['reinforce']:
                train_log.update({"Reward" : reward,
                "train R@1" : interaction.aux['batch_acc1'].item(),
                "reinforce" : interaction.aux['reinforce'].item(),
                "log_prob" : interaction.aux['log_prob'].mean().item()
            })

            if WANDB:
                wandb.log(train_log, step = self.STEP)
            self.STEP+=1
            if self.optimizer_scheduler:
                self.optimizer_scheduler.step()
            
            if config['mllm']=="llava-phi" and batch_id % config["iters_per_eval"] == 0 and batch_id != 0:
                torch.cuda.empty_cache()
                metric = self.rand_neg_val(epoch + 1, WANDB, config = config,  inference=False)
            # Saving model
            # if (self.SAVE_BEST_METRIC and metric > self.best_metric_score) or (self.opts.checkpoint_freq > 0 and (epoch + 1) % self.opts.checkpoint_freq==0): 
            #     for idx, callback in enumerate(self.callbacks):
            #         """
            #         callbacks.ConsoleLogger: aggregated_print
            #         finetuning.utils.ModelSaver: save_clipcap_model > {run_name}_e/final/best.pt                   
            #         callbacks.CheckpointSaver: pass
            #         """
            #         callback.on_epoch_end(train_loss, train_interaction, epoch + 1, config['WANDB']['run_name'], self.SAVE_BEST_METRIC)
                    
            #     if SAVE_BEST_METRIC:
            #         self.best_metric_score = metric


        mean_loss /= n_batches
        # full_interaction = Interaction.from_iterable(interactions)

        full_interaction  = defaultdict(list)
        for interaction in interactions:
            for k,v in interaction.aux.items():
                full_interaction[k].append(v.item())
        full_interaction  = {k: np.mean(v).mean() for k,v in full_interaction.items()}
        # print("****"*30)
        # # print(sum(p.mean() for p in self.game.sender.clip.visual.parameters()))
    
        # new_weights = {}
        # for name, param in self.game.sender.clipcap.gpt.transformer.named_parameters():
        #     new_weights[name] = param
        # # print(new_weights["h.0.attn.c_attn.parametrizations.weight.original"].requires_grad)
        # # print(f"og weight = {new_weights['h.0.attn.c_attn.parametrizations.weight.original'].mean()}")
        # print(f"lora weight = {new_weights['h.0.attn.c_proj.parametrizations.weight.0.lora_A'].mean()}")

        # print(f"LoRA param : {sum(p.mean() for p in self.game.sender.clipcap.gpt.transformer.h[0].attn.c_attn.parameters())}")
        # print(f"frozen param : {next(self.game.sender.clipcap.gpt.lm_head.parameters()).sum()}")
        # print(next(self.game.sender.clipcap.gpt.lm_head.parameters()).requires_grad)

        # print("****"*30)
        return mean_loss.item(), full_interaction

    def train(self,config, opts, inference = False):


        print(f"Total trainable params : {trainable_params(self.game.sender)}")
        count_trainable_parameters(self.game.sender)
        n_epochs = n_epochs = [_ for _ in config['neg_mining']['curricullum'].keys()][-1]
        WANDB = config['WANDB']['logging']
        if self.distributed_context.is_distributed:
            WANDB = WANDB and self.distributed_context.is_leader
        print(f"+++++++ WANDB  ={WANDB} LOCAL_RANK = {self.distributed_context.is_leader}  ++++++")
        self.INIT_VAL = config['INIT_VAL']
        self.GREEDY_BASELINE = config['GREEDY_BASELINE']
        self.SAVE_BEST_METRIC = config['SAVE_BEST_METRIC']
        self.train_method = config["train_method"]
        self.SAVE_USING_NEG = config["neg_mining"]["save_using_neg"] and config["neg_mining"]["do"]  
        self.best_metric_score = 0
        self.opts = opts


        inference_log_dir = os.path.join(config["inference"]["output_dir"].split("/inference")[0], "inference_log")
        if not os.path.isdir(inference_log_dir):
            os.makedirs(inference_log_dir)

        #INIT VAL
        if inference or (self.INIT_VAL and self.distributed_context.is_leader):
            metric = self.rand_neg_val(self.start_epoch , WANDB, config = config,  inference=inference)
        if inference:                
            return

        for callback in self.callbacks:
            """
            In CallBack class, create self.trainer = callbacks.console_logger , finetuning.utils.ModelSaver , callbacks.checkpointsaver
            """
            callback.on_train_begin(self)

        
        for epoch in range(self.start_epoch, n_epochs):

            if self.distributed_context.is_distributed:
                self.train_data.sampler.set_epoch(epoch)
                # self.validation_data.sampler.set_epoch(epoch)     

            # Train epoch
            if self.distributed_context.is_distributed:
                dist.barrier()
            print(f"Training epoch {epoch + 1}")

            # for callback in self.callbacks:
            #     callback.on_epoch_begin(epoch + 1)
            
            level_bsz = get_ds(epoch, config)                 
            # loader = get_loader(epoch, config['neg_mining']['curricullum'])
            print("***"*50)
            print(f"epoch :{epoch}")
            print(f"level_bsz : {level_bsz}")
            print("***"*50)
            train_loss, train_interaction = self.train_epoch(self.train_loaders[level_bsz], WANDB, self.GREEDY_BASELINE, self.train_method, self.opts, config, epoch)
            if WANDB:
                wandb.log({"Avg Loss" : train_loss,
                            "epoch" : epoch + 1}, step = self.STEP)

            if self.distributed_context.is_leader:
                torch.cuda.empty_cache()
                metric = self.rand_neg_val(epoch + 1, WANDB, config = config)
                
                # Saving model
                save_epoch = (self.opts.checkpoint_freq > 0 and (epoch + 1) % self.opts.checkpoint_freq==0)
                save_best = (self.SAVE_BEST_METRIC and metric > self.best_metric_score) 
                if save_best or save_epoch: 
                    for idx, callback in enumerate(self.callbacks):
                        """
                        callbacks.ConsoleLogger: aggregated_print
                        finetuning.utils.ModelSaver: save_clipcap_model > {run_name}_e/final/best.pt                   
                        callbacks.CheckpointSaver: pass
                        """
                        callback.on_epoch_end(train_loss, train_interaction, epoch + 1, config['WANDB']['run_name'], save_best, save_epoch)
                        
                    if self.SAVE_BEST_METRIC:
                        self.best_metric_score = metric


    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)
    
    def save_val_preds(self, full_interaction, config, inference = False):
        preds = [j.strip() for i in full_interaction["message"] for j in i]
        cocoids = [i for i in full_interaction['cocoid']]
        val_preds =  dict(zip(cocoids, preds))

        if inference:
            save_path = os.path.join(config["inference"]["output_dir"], f"{config['captions_type']}_{config['opts']['checkpoint_dir'].split('/')[-1]}.pkl")                                        
            # save_path = os.path.join(config["inference"]["output_dir"], f"{config['captions_type']}_mle.pkl")                                        
        else:    
            save_path = os.path.join(config["opts"]["checkpoint_dir"].split("checkpoints")[0] + "val_preds", config["WANDB"]["run_name"] + f"_val_preds.pkl")                                        
        
        # print("$$$$"*100)
        # print(save_path)
        if config['SAVE_PREDS']:
            with open(save_path, "wb") as f:
                pickle.dump(val_preds, f)
            
        eval_on_bags(config)


# print(f" wte norm : {torch.norm(self.game.sender.clipcap.gpt.transformer.wte.weight)}")   
# print(f" clip_proj norm : {torch.norm(self.game.sender.clipcap.clip_project.model[0].weight)}")   
# print(f" wte grad : {torch.norm(self.game.sender.clipcap.gpt.transformer.wte.weight.grad)}")    

#check if clip param changing 
# print(f"mlp :  {torch.norm(self.game.sender.clip.visual.transformer.resblocks[0].mlp.c_proj.weight)}")
# print(f"attn :  {torch.norm(self.game.sender.clip.visual.transformer.resblocks[0].attn.out_proj.weight)}")

# check if A.B in lora changing
# dummy = self.game.sender.clipcap.gpt.transformer.h[0].attn.c_attn.parametrizations.weight
# dummy = self.game.sender.clip.visual.transformer.resblocks[0].attn.parametrizations
# dummy = self.game.sender.clip.visual.transformer.resblocks[0].mlp.c_proj.parametrizations.weight

# x = [(name, p) for name, p in dummy.named_parameters()]
# print("***"*30)
# print(f"grad A : {torch.norm(x[1][-1])}")
# print(f"grad B : {torch.norm(x[2][-1])}")
# print(f"A : {torch.norm(x[1][-1])}")
# print(f"B : {torch.norm(x[2][-1])}")
# print(f"lora (A.B).norm() : {torch.norm(torch.matmul(x[1][-1].t(),x[2][-1].t()))}")
# # print(f"og weights  :{torch.norm(x[0][-1])}")
# print("***"*30)
