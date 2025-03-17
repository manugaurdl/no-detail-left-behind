import time
from typing import Callable
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from egg.core.baselines import MeanBaseline, NoBaseline
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.finetuning.clipcap import ClipCapSender, LLavaSender, LLavaPhi
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
                    captions : 5 coco GT cap for each image : list of 5 lists --> each sublist has bsz captions
                    }
        """
        
        CIDER_OPTIM = isinstance(self.loss, CiderReward)

        if CIDER_SR and self.training:

            if self.training:
                training = True
            else:
                training = False
            scaling_factor = 1.0

            # assert GREEDY_BASELINE, "Cider optim is done with greedy baseline"
            self.loss_sr = DiscriminativeLoss()
            self.loss
            aux_info = {}
            
            captions = None
            reward = None
            if reinforce or not self.training :
                if self.mllm=="clipcap":
                      captions, log_prob, kl_div= self.sender(sender_input, aux_input, use_fp16 = self.config['fp16']) # logprob : (B) --> only one logprob per caption (averaged over all words)
        
                with torch.no_grad():
                    
                    text_feats, img_feats = self.receiver(captions, receiver_input, aux_input) 
                    policy_acc, aux_info_disc_loss = self.loss_sr(text_feats, img_feats, self.training, True,  aux_input) # SR Reward
                    policy_acc = policy_acc.detach()

                    aux_info_disc_loss['acc_5'] = aux_info_disc_loss['acc_5'].mean()
                    aux_info_disc_loss['acc'] = aux_info_disc_loss['acc'].mean()
                    if not self.training:
                        aux_info_disc_loss['mean_rank'] = aux_info_disc_loss['mean_rank'].mean()
                        aux_info_disc_loss['median_rank'] = aux_info_disc_loss['median_rank'].float().mean()
                        aux_info_disc_loss['clip_s'] = aux_info_disc_loss['clip_s'].mean()

                    
                    aux_info.update(aux_info_disc_loss)

                    if GREEDY_BASELINE:
                        if training:
                            self.eval()
                        greedy_captions, _, kl_div = self.sender(sender_input, aux_input, greedy_baseline = True)
                        text_feats_greedy = self.receiver(greedy_captions, receiver_input, aux_input, img_feats = False) 
                        greedy_acc, _ = self.loss_sr(text_feats_greedy, img_feats, False, True,  aux_input)  # GREEDY reward
                        greedy_acc = greedy_acc.detach()
                        greedy_cider = -1 * (torch.tensor(self.loss(greedy_captions, aux_input)).to(log_prob.device).detach())
                        
                        baseline = greedy_acc + scaling_factor * greedy_cider
                        if training:
                            self.train()
                    

                #Get CIDer 
                policy_cider = -1 * (torch.tensor(self.loss(captions, aux_input)).to(log_prob.device).detach())

                policy_reward = policy_acc + scaling_factor * policy_cider 
                print(f"------" * 30)
                print(f"R@1 : {policy_acc.mean()}")
                print(f"CIDEr: {policy_cider.mean()}")
                print(f"------" * 30)
                
                weighted_kl_div = self.kl_div_coeff * kl_div
                
                if not GREEDY_BASELINE:
                    baseline = self.baseline.predict(policy_reward.detach())
                    if self.training:
                        self.baseline.update(policy_reward)

                reward = (policy_reward - baseline)

                batch_acc1 = aux_info['acc'].mean()
                
                loss = (reward * log_prob).mean()

                aux_info["reinforce"] = loss.detach()
                aux_info["kl_div"] = kl_div
                aux_info["log_prob"] =  log_prob.detach().mean()
                aux_info["batch_acc1"] = batch_acc1
                reward = reward.mean().item()
            
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


        ################################################################################################################################################################
        
        elif CIDER_OPTIM and not inference and not CIDER_SR:

            if self.training:
                training = True
            else:
                training = False            
            # get policy cap
            policy_captions, log_prob, kl_div = self.sender(sender_input, aux_input) # logprob : (B) --> only one logprob per caption (averaged over all words)
        
            #get greedy_cap
            if GREEDY_BASELINE:
                if training:
                    self.eval()
                with torch.no_grad():
                    greedy_cap, _, _  = self.sender(sender_input, aux_input, False, greedy_baseline = GREEDY_BASELINE)
                    baseline = torch.tensor(self.loss(greedy_cap, aux_input)).to(log_prob.device).detach()
                if training:
                    self.train()

            # gt : aux_input
            # baseline --> mean vs greedy
            policy_cider = torch.tensor(self.loss(policy_captions, aux_input)).to(log_prob.device)
                        
            weighted_kl_div = self.kl_div_coeff * kl_div
            
            if not GREEDY_BASELINE:
                # get policy cap first.
                baseline = self.baseline.predict(policy_cider.detach())
            reward = (policy_cider.detach() - baseline) #+ weighted_kl_div            
            loss = -1* ((reward * log_prob).mean())

            # import ipdb;ipdb.set_trace()
            if self.training and not GREEDY_BASELINE:
                self.baseline.update(policy_cider)

            aux_info = {'batch_acc1' : torch.randn(1,2).mean(), "kl_div" : kl_div, "log_prob" : log_prob.detach().mean()}
            aux_info['reinforce'] = loss.detach().mean() 
            
            logging_strategy = self.test_logging_strategy

            interaction = logging_strategy.filtered_interaction(
                sender_input=sender_input,
                labels=cocoids,
                receiver_input=receiver_input,
                aux_input=aux_input,
                message=policy_captions,
                receiver_output=None,
                message_length=None,
                aux=aux_info,
                )
        
        elif isinstance(self.loss, DiscriminativeLoss) or not self.training:
            
            training = self.training

            aux_info = {}
            if not self.training:
                self.loss_old = self.loss
                self.loss = DiscriminativeLoss()

            training = self.training
            #llava clip -> self.sender.model.model.vision_tower.vision_tower.vision_model
            #CLIP zero shot baseline

            if self.mllm == "clipcap" and contrastive:    
                sender_img_feats = self.sender.clip.encode_image(receiver_input)
                sender_img_feats = sender_img_feats / sender_img_feats.norm(dim=1, keepdim=True)

                gt_caps_tokens = clip.tokenize(aux_input['captions'][0], truncate=True).to(receiver_input.device)
                sender_text_feats = self.sender.clip.encode_text(gt_caps_tokens)
                sender_text_feats = sender_text_feats / sender_text_feats.norm(dim=1, keepdim=True)
                
                if not self.training:
                    _ , clip_zero_shot = self.loss(sender_text_feats, sender_img_feats, self.training, True,  aux_input)

                if contrastive:
                    logits_per_image = self.sender.clip.logit_scale * sender_img_feats @ sender_text_feats.t()
                    logits_per_text = logits_per_image.t()

                    label_id = torch.arange(receiver_input.shape[0]).cuda(non_blocking=True)
                    loss_e2t = F.cross_entropy(logits_per_image, label_id)
                    loss_t2e = F.cross_entropy(logits_per_text, label_id)
                    contrastive_loss = loss_e2t + loss_t2e
                    aux_info["contrastive"] = contrastive_loss.detach()
            
            captions = None
            reward = None
            if reinforce or not self.training :
                if self.mllm=="clipcap" and len(receiver_input.shape) !=5 :
                      captions, log_prob, kl_div= self.sender(sender_input, aux_input, use_fp16 = self.config['fp16']) # logprob : (B) --> only one logprob per caption (averaged over all words)
                elif self.mllm=="llava-phi":
                    captions, log_prob, kl_div, receiver_input = self.sender(sender_input, aux_input, use_fp16 = self.config['fp16']) # logprob : (B) --> only one logprob per caption (averaged over all words)

                with torch.no_grad():

                    if not isinstance(receiver_input, torch.Tensor):
                        receiver_input = torch.stack([transform(recv_inp) for recv_inp in receiver_input]).to(next(iter(self.receiver.parameters())))
                    
                    ################################## IMAGE CODE EVAL  ##################################################
                    
                    if len(receiver_input.shape)==5:
                        
                        if captions is None:
                            ###VLM Eval 
                            # sender clip 
                            
                            text_feats, img_feats = self.get_sender_clip_feats(aux_input['captions'], receiver_input.view(-1,3,224,224), aux_input)

                            # text_feats, img_feats = self.receiver(aux_input['captions'], receiver_input.view(-1,3,224,224), aux_input) 
                            img_feats = img_feats.view(-1, 10, text_feats.shape[-1]) # img : B, 10, 512
                            text_feats = text_feats.unsqueeze(1) # text : B, 1 ,512
                            
                            acc_1 = []
                            acc_5 = []
                            clip_s = []
                            mean_rank = []
                            median_rank = []

                            for i in range(text_feats.shape[0]):
                                sr_loss, aux_info_disc_loss = self.loss(text_feats[i],  img_feats[i], self.training, True,  aux_input)

                                acc_1.append(aux_info_disc_loss['acc'])
                                
                                acc_5.append(aux_info_disc_loss['acc_5'])
                                clip_s.append(aux_info_disc_loss['clip_s'])
                                mean_rank.append(aux_info_disc_loss['mean_rank'])
                                median_rank.append(aux_info_disc_loss['median_rank'])

                            aux_info_disc_loss['acc_5'] = torch.stack(acc_5).mean()
                            aux_info_disc_loss['acc'] = torch.stack(acc_1).mean()
                            aux_info_disc_loss['clip_s'] = torch.stack(clip_s)
                            aux_info_disc_loss['median_rank'] = torch.stack(median_rank)
                            aux_info_disc_loss['mean_rank'] = torch.stack(mean_rank)

                            return aux_info_disc_loss
                    else:
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
                        if training :
                            self.eval()
                        greedy_captions, _, kl_div = self.sender(sender_input, aux_input, greedy_baseline = True)
                        text_feats_greedy = self.receiver(greedy_captions, receiver_input, aux_input, img_feats = False) #clip_feats
                        baseline, _ = self.loss(text_feats_greedy, img_feats, self.training, True,  aux_input)
                        baseline = baseline.detach()
                        if training:
                            self.train()
                    
                    # text_feats = self.receiver(captions, receiver_input, aux_input) #clip_feats
                    # loss, aux_info = self.loss(text_feats, receiver_input.squeeze(1), self.training, True,  aux_input)
                weighted_kl_div = self.kl_div_coeff * kl_div
                
                if not GREEDY_BASELINE:
                    baseline = self.baseline.predict(sr_loss.detach())
                    if self.training:
                        self.baseline.update(sr_loss)

                batch_acc1 = aux_info['acc'].mean()
                
                reward = (sr_loss.detach() - baseline)# + weighted_kl_div
                reinforce_loss = (reward * log_prob).mean()
                aux_info["reinforce"] = reinforce_loss.detach()
                aux_info["kl_div"] = kl_div
                aux_info["log_prob"] =  log_prob.detach().mean()
                aux_info["batch_acc1"] = batch_acc1
                reward = reward.mean().item()
            
            if not self.training and contrastive:
                aux_info["recall_5_clip_zs"] = clip_zero_shot['acc_5'].mean()
                aux_info["recall_1_clip_zs"] = clip_zero_shot['acc'].mean()

            if contrastive and reinforce:
                loss = reinforce_loss * 100 + contrastive_loss
            elif not contrastive:
                loss = reinforce_loss  
            elif not reinforce:
                loss = contrastive_loss
                
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

        else:
            outputs = self.sender(sender_input, aux_input, train_method= train_method)
            if self.training:
                targets = aux_input['tokens'].view(-1, aux_input["tokens"].shape[-1])
                mask = aux_input['mask'].view(-1, aux_input["mask"].shape[-1])
                # targets, mask = targets.to(device), mask.to(device)
                logits = outputs.logits[:, self.prefix_len - 1: -1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(torch.long).flatten(), ignore_index=0) # (B,T) flattened to (B*T)
                # probs = torch.nn.functional.softmax(logits, dim=-1)
                # preds = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1) # preds is flattened out --> (B*max_cap_len , 1)
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


            # weighted_kl_div = self.kl_div_coeff * kl_div
            # aux_info["kl_div"] = weighted_kl_div
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
    if config['mllm'] == "clipcap":
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
    elif config['mllm'] == "llava":
        sender = LLavaSender(
            # clip_model=opts.sender_clip_model,
            # clipcap_path=opts.mle_model_path,
            # official_clipcap_weights = config["official_clipcap_weights"],
            train_method= config["train_method"],
            config=config,
            do_sample=opts.do_sample,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
        )
    elif config['mllm'] == "llava-phi":
        sender = LLavaPhi(
            train_method= config["train_method"],
            config=config,
            do_sample=opts.do_sample,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
        )
    else:
        raise RuntimeError("selected unsupported MLLM in config")

    # assert opts.recv_clip_model =="ViT-L/14@336px"
    receiver = ClipReceiver(clip_model=opts.recv_clip_model)
    receiver.clip.eval()
    for p in receiver.clip.parameters():
        p.requires_grad = False
    # if loading clip lazy feats
    # receiver.clip.visual = None
    # torch.cuda.empty_cache()
    
    test_logging_strategy = LoggingStrategy(
        False, False, True, True, True, False, False
    )

    loss_fn = get_loss(
        loss_type=opts.loss_type,
        dataset=opts.train_dataset,
        num_hard_negatives=opts.num_hard_negatives,
    )

    if config['mllm']=="clipcap" and config["lora"]:

        # original_weights = {}
        # for name, param in sender.clipcap.gpt.transformer.named_parameters():
        #     original_weights[name] = param.clone().detach()
        # with open("/home/manugaur/temp/gpt_transformer_og.pkl", "wb") as f:
        #     pickle.dump(original_weights, f)

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
