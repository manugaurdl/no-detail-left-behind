# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.zoo.emergent_captioner.utils import DATASET2NEG_PATHS
from egg.zoo.emergent_captioner.evaluation.evaluate_nlg import compute_nlg_metrics


class Loss(nn.Module):
    def __init__(
        self,
        train_emb_path: str = None,
        train_nns_path: str = None,
        num_hard_negatives: int = 0,
    ):
        super().__init__()



        self.train_emb = None
        self.train_nns = None
        train_emb_path = None
        if train_emb_path:
            assert train_nns_path
            self.emb = torch.load(train_emb_path, map_location="cpu")
            self.nns = torch.load(train_nns_path, map_location="cpu")

        self.num_hard_negatives = num_hard_negatives

    def get_similarity_scores(self, text_feats, image_feats, training, aux_input=None):
        cosine_in_batch = text_feats @ image_feats.t() # (B x B) sim matrix
        
        targets = cosine_in_batch.diag(0).unsqueeze(1) # targets are the image itself --> extract the main diagnol (B x 1)
        cosine_in_batch.fill_diagonal_(float("-inf"))  # mask targets.
        cosine_in_batch = torch.cat([targets, cosine_in_batch], dim=1) # B x (1+B)

        cosine_sims = cosine_in_batch

        if self.num_hard_negatives > 0 and self.nns:
            elem_idxs = img_idxs.squeeze()

            # fetches embeddings of nearest-neighbor hard negatives
            batch_nns = self.nns[elem_idxs][:, 1 : self.num_hard_negatives + 1].long()

            # batch x num_negatives x embed_dim
            image_feats_negatives = self.emb[batch_nns].to(text_feats.device)[1:]

            cosine_negatives = torch.einsum(
                "be,bne->bn", text_feats, image_feats_negatives
            )

            cosine_sims = torch.cat([cosine_in_batch, cosine_negatives], dim=1)
        
        if aux_input is not None:
            aux_input["receiver_output"] = cosine_sims.detach()
        
        return cosine_sims

    def remove_fields_negatives(self):
        self.nns = None
        self.emb = None

    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        raise NotImplementedError


class DiscriminativeLoss(Loss):
    def forward(self, text_feats, img_feats, training, get_acc_5, aux_input):

        sims = self.get_similarity_scores(text_feats, img_feats, training, aux_input=aux_input) # Sim matrix of size [B | B X B]. First column = targets of self retrieval

        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)
        # dist over bsz classes. Even though tensor of shape (bsz + 1), -inf values won't count due to softmax. 
        # logprob for target class in the 0th column. For each sample(row),logprob of target class : row[0]

        loss = F.cross_entropy(sims, labels, reduction="none")
        # For each row, highest value should be the first colum i.e argmax for each row in sims should be == 0.
        
        #argmax --> recall@1 
        out = {}
        acc_1 = (sims.argmax(dim=1) == labels).detach().float() 
        out["acc"]= acc_1

        if get_acc_5:      
            top_5 = torch.topk(sims, 5, dim = 1)[1].detach()
            acc_5 = torch.any(top_5 ==0, dim = 1).detach().float()
            out["acc_5"]= acc_5
        
        if training:
            return loss, out
        
        rank = torch.where(torch.argsort(-sims, dim =1) == 0)[1]
        rank+=1 # rank is indexed from 1 
        mean_rank = rank.float().mean()
        median_rank = rank.median()
        out["mean_rank"] = mean_rank
        out["median_rank"] = median_rank
        clip_s = torch.clamp(sims[:, 0]*100, min = 0)
        out["clip_s"] =clip_s
        
        return loss, out

class CiderReward(Loss):
    def forward(self, preds, aux_input):
        img_ids = aux_input['cocoid']
        captions = aux_input['captions']
        # preds_per_batch = full_interaction.message
        bsz = len(aux_input['cocoid'])

        coco_caps = list(zip(*captions))

        gold_standard = {idx : [{"caption": cap} for cap in img] for idx, img in enumerate(coco_caps)}

        predictions = {idx : [{"caption" :pred}] for idx, pred in enumerate(preds)}

        summary = compute_nlg_metrics(predictions, gold_standard, only_cider = True) # score for each idx stored in summary except bleu

        return summary['CIDEr']

class CEloss(Loss):
    def forward(self, preds, aux_input):
      pass

class AccuracyLoss(Loss):
    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        sims = self.get_similarity_scores(text_feats, img_feats, img_idxs, aux_input)

        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)

        acc = (sims.argmax(dim=1) == labels).detach().float()

        return -acc, {"acc": acc}


class SimilarityLoss(Loss):
    def forward(self, text_feats, img_feats, img_idxs, aux_input=None):
        sims = self.get_similarity_scores(text_feats, img_feats, img_idxs, aux_input)

        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)

        loss = -sims[:, 0]
        acc = (sims.argmax(dim=1) == labels).detach().float()

        return loss, {"acc": acc}


def get_loss(loss_type: str, dataset: str, num_hard_negatives: int):
    train_emb, train_nns = DATASET2NEG_PATHS.get(dataset.lower(), (None, None))

    name2loss = {
        "discriminative": DiscriminativeLoss,
        "accuracy": AccuracyLoss,
        "similarity": SimilarityLoss,
        "cider" : CiderReward,
        "mle": CEloss
        
    }

    loss_cls = name2loss.get(loss_type.lower(), None)
    assert loss_cls, f"cannot recognize loss {loss_type}"

    return loss_cls(train_emb, train_nns, num_hard_negatives)
