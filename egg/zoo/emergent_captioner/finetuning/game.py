import time
from typing import Callable
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from egg.core.baselines import MeanBaseline, NoBaseline
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.finetuning.clipcap import ClipCapSender
from egg.zoo.emergent_captioner.finetuning.losses import get_loss, CiderReward, DiscriminativeLoss
from egg.zoo.emergent_captioner.finetuning.receiver import ClipReceiver
from egg.zoo.emergent_captioner.finetuning.sender_clip import ClipSender

from egg.zoo.emergent_captioner.finetuning.lora import LoRA
import pickle
import clip
from egg.zoo.emergent_captioner.dataloaders import get_transform
from egg.zoo.emergent_captioner.finetuning.utils import int2mil, trainable_params

transform = get_transform(224, None)

class ReinforceCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        kl_div_coeff: float = 0.0,
        baseline: str = "no",
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
        config : dict = None,
    ):
        super(ReinforceCaptionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.baseline_name = baseline
        self.baseline = {"no": NoBaseline, "mean": MeanBaseline}[baseline]()

        self.kl_div_coeff = kl_div_coeff

        self.train_logging_strategy = (
            LoggingStrategy().minimal()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy().minimal()
            if test_logging_strategy is None
            else test_logging_strategy
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_len = config['prefix_len']
        self.mllm = config['mllm']
        self.config = config
        self.get_sender_clip_feats = ClipSender(self.sender.clip)

    def forward(self, sender_input, cocoids ,receiver_input=None, aux_input=None, GREEDY_BASELINE = False, train_method= None, inference = False, contrastive = False, reinforce = True, CIDER_SR = False):
        """
        kl_div_coeff : 0
        _____
        sender_input : (B, 3, 224, 224)
        labels : ? 
        receiver_input : (B, 3 ,224, 224)
        aux : dict {img_id : cocoids
                    captions : 5 coco GT cap for each image : list of 5 lists --> each sublist has bsz captions}
        """
        
        CIDER_OPTIM = isinstance(self.loss, CiderReward)
                
        if isinstance(self.loss, DiscriminativeLoss) or not self.training:

            aux_info = {}

            #preserve initial training and loss config 
            if not self.training:
                self.loss_old = self.loss
                self.loss = DiscriminativeLoss()
            training = self.training

            captions = None
            reward = None
            
            captions, log_prob, kl_div= self.sender(sender_input, aux_input, use_fp16 = self.config['fp16']) #one logprob for each caption (averaged over all words)

            with torch.no_grad():

                if not isinstance(receiver_input, torch.Tensor):
                    receiver_input = torch.stack([transform(recv_inp) for recv_inp in receiver_input]).to(next(iter(self.receiver.parameters())))
                
                text_feats, img_feats = self.receiver(captions, receiver_input, aux_input) #clip_feats
                sr_loss, aux_info_disc_loss = self.loss(text_feats, img_feats, self.training, True,  aux_input)
            
                aux_info_disc_loss['acc_5'] = aux_info_disc_loss['acc_5'].mean()
                aux_info_disc_loss['acc'] = aux_info_disc_loss['acc'].mean()
                
                if not self.training:
                    aux_info_disc_loss['mean_rank'] = aux_info_disc_loss['mean_rank'].mean()
                    aux_info_disc_loss['median_rank'] = aux_info_disc_loss['median_rank'].float().mean()
                    aux_info_disc_loss['clip_s'] = aux_info_disc_loss['clip_s'].mean()

                
                aux_info.update(aux_info_disc_loss)
                #R@1 from GREEDY dist
                if GREEDY_BASELINE:
                    if training:
                        self.eval()
                    
                    greedy_captions, _, kl_div = self.sender(sender_input, aux_input, greedy_baseline = True)
                    text_feats_greedy = self.receiver(greedy_captions, receiver_input, aux_input, img_feats = False) #clip_feats
                    baseline, _ = self.loss(text_feats_greedy, img_feats, self.training, True,  aux_input)
                    baseline = baseline.detach()
                    
                    if training:
                        self.train()
                
                else:
                    baseline = self.baseline.predict(sr_loss.detach())
                    if self.training:
                        self.baseline.update(sr_loss)
                
            weighted_kl_div = self.kl_div_coeff * kl_div
            batch_acc1 = aux_info['acc'].mean()
            
            reward = (sr_loss.detach() - baseline)# + weighted_kl_div
            reinforce_loss = (reward * log_prob).mean()
            aux_info["reinforce"] = reinforce_loss.detach()
            aux_info["kl_div"] = kl_div
            aux_info["log_prob"] =  log_prob.detach().mean()
            aux_info["batch_acc1"] = batch_acc1
            reward = reward.mean().item()
            loss = reinforce_loss
            
            logging_strategy = (
                self.train_logging_strategy if self.training else self.test_logging_strategy
            )
            interaction = logging_strategy.filtered_interaction(
                sender_input=sender_input,
                labels=cocoids,
                receiver_input=receiver_input,
                aux_input=aux_input,
                message=captions,
                receiver_output=None,
                message_length=None,
                aux=aux_info,
            )
            if not self.training:
                self.loss = self.loss_old

        ############# MLE ############
        else:
            outputs = self.sender(sender_input, aux_input, train_method= train_method)
            if self.training:
                targets = aux_input['tokens'].view(-1, aux_input["tokens"].shape[-1])
                mask = aux_input['mask'].view(-1, aux_input["mask"].shape[-1])
                logits = outputs.logits[:, self.prefix_len - 1: -1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(torch.long).flatten(), ignore_index=0) # (B,T) flattened to (B*T)
                baseline = self.baseline.predict(loss.detach())
                self.baseline.update(loss)
            else:
                val_captions, log_prob, kl_div = outputs
                aux_info = {"kl_div" :kl_div}
                aux_info["log_prob"] =  log_prob.detach()

                logging_strategy = (
                    self.train_logging_strategy if self.training else self.test_logging_strategy
                )
                interaction = logging_strategy.filtered_interaction(
                    sender_input=sender_input,
                    labels=cocoids,
                    receiver_input=receiver_input,
                    aux_input=aux_input,
                    message=val_captions,
                    receiver_output=None,
                    message_length=None,
                    aux=aux_info,
                )
                return torch.randn(1), interaction, torch.randn(1)

            aux_info = {}
            captions = []
            logging_strategy = (
                self.train_logging_strategy if self.training else self.test_logging_strategy
            )
            interaction = logging_strategy.filtered_interaction(
                sender_input=sender_input,
                labels=cocoids,
                receiver_input=receiver_input,
                aux_input=aux_input,
                message=captions,
                receiver_output=None,
                message_length=None,
                aux=aux_info,
            )
            
            return loss, interaction, loss.item()

        return loss, interaction, reward


def build_game(opts, config):

    sender = ClipCapSender(
        clip_model=opts.sender_clip_model,
        clipcap_path=opts.mle_model_path,
        official_clipcap_weights = config["official_clipcap_weights"],
        train_method= config["train_method"],
        config=config,
        do_sample=opts.do_sample,
        beam_size=opts.beam_size,
        max_len=opts.max_len,
    )

    # assert opts.recv_clip_model =="ViT-L/14@336px"
    receiver = ClipReceiver(clip_model=opts.recv_clip_model)
    receiver.clip.eval()
    for p in receiver.clip.parameters():
        p.requires_grad = False
    
    test_logging_strategy = LoggingStrategy(
        False, False, True, True, True, False, False
    )

    loss_fn = get_loss(
        loss_type=opts.loss_type,
        dataset=opts.train_dataset,
        num_hard_negatives=opts.num_hard_negatives,
    )

    if config["lora"]:
        LoRA(sender,config["lora_rank"], config['finetune_model'], config)
    
    if config['freeze_adapter']:
        for p in sender.clipcap.clip_project.parameters():
            p.requires_grad = False
    
    if config['freeze_wte']:
        for p in sender.clipcap.gpt.lm_head.parameters():
            p.requires_grad = False
    
    # if config['freeze_adapter']:
    #     for name, p in sender.clipcap.clip_project.named_parameters():
    #         p.requires_grad = False    

    # elif config['mllm']=="llava":
    #     for p in sender.model.model.parameters():
    #         p.requires_grad = False
        
    #     for p in sender.model.lm_head.parameters():
    #         p.requires_grad = True
        
    # elif config['mllm']=="llava-phi":
        
    #     for p in sender.model.parameters():
    #         p.requires_grad = False

    #     # for p in sender.model.multi_modal_projector.parameters():
    #     #     p.requires_grad = True

    #     for p in sender.model.language_model.lm_head.parameters():
    #         p.requires_grad = True


    game = ReinforceCaptionGame(
        sender=sender,
        receiver=receiver,
        loss=loss_fn,
        baseline=opts.baseline,
        kl_div_coeff=opts.kl_div_coeff,
        test_logging_strategy=test_logging_strategy,
        config = config,
    )
    return game
