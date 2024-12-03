import torch
import torch.nn as nn
import torch.nn.functional as F
from egg.zoo.emergent_captioner.finetuning.utils import int2mil, trainable_params
import torch.nn.utils.parametrize as parametrize


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha, device = None):
        super().__init__()
        # device : same device as feats
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        # Section 4.1 of the paper: 
        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. 
        #   As a result, we simply set α to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        # self.enabled = True

    def forward(self, original_weights):
        # if self.enabled:
        # Return W + (B*A)*scale
        return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        # else:
        #     return original_weights


def linear_layer_parameterization(layer, device, rank, lora_alpha=16):
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )

def inproj_parameterization(layer, device, rank, lora_alpha):    
    features_in, features_out = layer.in_proj_weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )

def parameterize(layer,weight_name, rank):
    parametrize.register_parametrization(
layer, weight_name, linear_layer_parameterization(layer, layer.weight.device, rank), unsafe =True)

def LoRA(model,  rank, model_type, config):
    
    if model_type =="llm" or model_type=="both":
        
        print(f"trainable params before LORA :{trainable_params(model)}")
        
        if config['mllm']=="clipcap":
        
            for i in range(12):
                parameterize(model.clipcap.gpt.transformer.h[i].attn.c_attn, "weight", rank)
                parameterize(model.clipcap.gpt.transformer.h[i].attn.c_proj, "weight", rank)
                # parameterize(model.clipcap.gpt.transformer.h[i].mlp.c_fc, "weight", rank)
                # parameterize(model.clipcap.gpt.transformer.h[i].mlp.c_proj, "weight", rank)
            
            # freeze GPT
            for name, param in model.clipcap.gpt.named_parameters():
                if 'lora' in name : #or "wte.weight" in name:
                    continue
                else:
                    param.requires_grad = False
        
        elif config['mllm']=="llava-phi":
            for i in range(32):
                
                parameterize(model.model.language_model.model.layers[i].self_attn.q_proj, "weight", rank)
                parameterize(model.model.language_model.model.layers[i].self_attn.k_proj, "weight", rank)
                parameterize(model.model.language_model.model.layers[i].self_attn.v_proj, "weight", rank)
                parameterize(model.model.language_model.model.layers[i].self_attn.o_proj, "weight", rank)
                parameterize(model.model.language_model.model.layers[i].mlp.gate_proj, "weight", rank)
                parameterize(model.model.language_model.model.layers[i].mlp.up_proj, "weight", rank)
                parameterize(model.model.language_model.model.layers[i].mlp.down_proj, "weight", rank)
                # parameterize(model.model.language_model.lm_head, "weight", rank)


            for name, param in model.named_parameters():
                if 'lora' in name:
                    continue
                else:
                    param.requires_grad = False

        print(f"trainable params after LORA :{trainable_params(model)}")
    
    if model_type=="clip" or model_type=="both":
        print(f"CLIP trainable params before LORA :{trainable_params(model.clip)}")
        for l_num in range(12):
            parametrize.register_parametrization(
                model.clip.visual.transformer.resblocks[l_num].attn,
                "in_proj_weight", 
                inproj_parameterization(model.clip.visual.transformer.resblocks[l_num].attn,
                next(model.clip.visual.parameters()).device,
                rank=rank,
                lora_alpha=16)
                        )
        if config['clip_mlp_ft']:
            for l_num in range(12):
                parameterize(model.clip.visual.transformer.resblocks[l_num].mlp.c_fc, "weight", rank)
                parameterize(model.clip.visual.transformer.resblocks[l_num].mlp.c_proj, "weight", rank)

        if config["clip_text"]:
            for l_num in range(12):
                parametrize.register_parametrization(
                        clip.transformer.resblocks[l_num].attn,
                        "in_proj_weight",
                        inproj_parameterization(clip.transformer.resblocks[l_num].attn,
                        next(clip.transformer.parameters()),
                        rank=rank,
                        lora_alpha=16)
                )

                parameterize(clip.transformer.resblocks[l_num].attn.out_proj, "weight", rank=rank)
            #freeze params
        if config['adapter_lora']:
            parameterize(model.clipcap.clip_project.model[0], "weight", config["adapter_rank"])
            parameterize(model.clipcap.clip_project.model[2], "weight", config["adapter_rank"])
            
            for name, param in model.clipcap.clip_project.named_parameters():
                if 'lora' in name:
                    continue
                else:
                    param.requires_grad = False

        for name, param in model.named_parameters():
            if "clip_project" in name or 'lora' in name:#or 'wte.weight' in name: 
                continue
            else:
                param.requires_grad = False            
    
        print(f"CLIP trainable params before LORA :{trainable_params(model.clip)}")

