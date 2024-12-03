import csv
import os
from PIL import Image
import torch
from clip import load
import clip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm

from egg.zoo.emergent_captioner.dataloaders import get_transform
from transformers import GPT2Tokenizer

clipcap_transform = get_transform(224, None)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_len = 25 # MLLM : 25 (darshan's caps) | VLM : 11, 15 (a photo of )
rand_25 = False
def get_probs(model, images, token_list):

    all_tokens = []
    all_masks = []
    for tokens in token_list:  
        padding = max_len - len(tokens)
        pad = torch.zeros(padding)
        mask = torch.cat((torch.ones(10+len(tokens)), pad)).cuda()
        tokens = torch.cat((tokens, pad)).int()
        all_tokens.append(tokens)
        all_masks.append(mask)
    tokens = torch.stack(all_tokens, dim = 0)
    mask = torch.stack(all_masks, dim = 0)
    images = images.to(next(iter(model.clipcap.clip_project.parameters())).device)
    tokens = tokens.to(next(iter(model.clipcap.clip_project.parameters())).device)
    image_feats = model.clip.visual(images.unsqueeze(0))
    prompts = model.clipcap.clip_project(image_feats) #16,7680
    prompts = prompts.view(image_feats.shape[0], 10, -1) # 16,10,768
    prompts = prompts.repeat(2, 1, 1)
    bsz, prefix_len, h_dim = prompts.shape
    tokens_flat = tokens.view(-1,tokens.shape[-1]) # B*5, 40
    token_emb = model.clipcap.gpt.transformer.wte(tokens_flat) #B*5, 40 , 768
    gpt_input = torch.cat((prompts, token_emb), dim = 1) # B*5, 50, 768
    mask = mask.view(-1, mask.shape[-1]) # B*5, 50
    out = model.clipcap.gpt(inputs_embeds = gpt_input, attention_mask = mask)
    probs = out.logits[:, 9: -1, :len(gpt2_tokenizer)].softmax(dim=-1).cpu()[: , :, :] 
    probs = torch.gather(probs, dim = -1, index = tokens.unsqueeze(2).to(torch.int64).cpu()).squeeze(-1)
    probs*= mask[:, 10:].cpu()
    return probs.mean(dim=-1)





def mmvp_mllm_benchmark(model, preprocess, benchmark_dir, device = "cpu"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = load(model_name, device=device)
    image_dir = os.path.join(benchmark_dir, 'images')
    txt_file = os.path.join(benchmark_dir, 'mmvp_captions.txt')
    

    # csv_outfile = open('output.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_outfile)
    # csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    # categories = [
    #     'Orientation and Direction', 'Presence of Specific Features', 
    #     'State and Condition', 'Quantity and Count', 
    #     'Positional and Relational Context', 'Color and Appearance',
    #     'Structural Characteristics', 'Texts',
    #     'Viewpoint and Perspective'
    # ]

    # pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    c= 0
    score = []
    with open(txt_file, 'r') as f:
        for i, row in enumerate(f):
        # next(reader)  # skip header
        # for i, row in enumerate(reader):

            # continue
            # qid1, qtype1, statement1 = row
            
            qid1, text1 , _ = row.split(".")
            # Get next row for the pair
            row = next(f)
            qid2, text2 , _ = row.split(".")
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img = Image.open(os.path.join(image_dir, f'{qid1}.jpg')).convert("RGB")
            # img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg')).convert("RGB")
        
            img = clipcap_transform(img)
            # img2 = clipcap_transform(img2)

            # text1 = 'a photo of ' + statement1
            # text2 = 'a photo of ' + statement2

            t1_tokens = torch.tensor(gpt2_tokenizer.encode(text1),dtype=torch.int)
            t2_tokens = torch.tensor(gpt2_tokenizer.encode(text2),dtype=torch.int)
            
            i1_p1, i1_p2 = get_probs(model, img, [t1_tokens, t2_tokens]) #p1>p2
            # i2_p1, i2_p2 = get_probs(model, img2, [t1_tokens, t2_tokens]) #p2>p1
            
            if c%2==0:
                score.append(int(i1_p1 > i1_p2))
            else:
                score.append(int(i1_p1 < i1_p2))
            

            # current_category = categories[num_pairs // 15]
            # if rand_25 :
            #     if i1_p1.item() > i1_p2.item() and  i2_p1.item() < i2_p2.item():
            #         pair_accuracies[current_category] += 1
            #         num_pairs += 1

            # else: 
            #     if i1_p1.item() > i1_p2.item():
            #         pair_accuracies[current_category] += 1
                
            #     if  i2_p1.item() < i2_p2.item():
            #         pair_accuracies[current_category] += 1
                
            #     num_pairs += 2

            c+=1
            #########################################################################################################
            # text1 = clip.tokenize([text1]).to(device)
            # text2 = clip.tokenize([text2]).to(device)
            
            # img1 = preprocess(img1).unsqueeze(0).to(device)
            # img2 = preprocess(img2).unsqueeze(0).to(device)
            # imgs = torch.cat((img1, img2), dim=0)


            # with torch.no_grad():
            #     logits_per_image1, logits_per_text1 = model(imgs, text1)
            #     logits_per_image2, logits_per_text2 = model(imgs, text2)
                
            #     probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
            #     probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            # img1_score1 = probs1[0][0]
            # img1_score2 = probs2[0][0]
            
            # pred1 = "img1" if img1_score1 > 0.5 else "img2"
            # pred2 = "img1" if img1_score2 > 0.5 else "img2"

            # gt1 = "img1" if qid1 % 2 == 1 else "img2"
            # gt2 = "img1" if qid2 % 2 == 1 else "img2"

            
            # csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            # current_category = categories[num_pairs // 15]
            # if pred1 == gt1 and pred2 == gt2:
            #     pair_accuracies[current_category] += 1
            # num_pairs += 1

        # csv_outfile.close()

    # Calculate percentage accuracies
    # for category in pair_accuracies:
    #     pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    if rand_25:
        out_score = np.array(score).mean()
    else: 
        pairs = []
        for i in range(0, len(score), 2):
            if i+1 < len(score):
                pairs.append((score[i], score[i+1]))

        out_score = (np.array([1 for pair in pairs if 0 not in pair]).sum())/len(pairs)

    return pair_accuracies


# parser = argparse.ArgumentParser(description='Process a directory path.')
    
# # Adding an argument for the directory path
# parser.add_argument('--directory', type=str, help='The path to the directory')

# # Parsing the arguments
# args = parser.parse_args()

# # OpenAI models
# models = ['ViT-L/14']

# results_openai = {f'openai-{model}': benchmark_model(model, args.directory) for model in models}


# Merge results
# results = {**results_openai}

# # Convert results to format suitable for star plot
# categories = results[list(results.keys())[0]].keys()
# data = {'Categories': list(categories)}
# for model in list(results_openai.keys()):
#     data[model] = [results[model][category] for category in categories]

# print(results)
