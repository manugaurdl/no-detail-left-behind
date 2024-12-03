import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import clip



dataset  = load_dataset('facebook/winoground', use_auth_token="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")

rand_25 = False


def winoground_vlm(model, preprocess):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {}
    img_scores = []
    text_scores = []

    for idx, sample in tqdm(enumerate(dataset['test']), total = len(dataset['test'])):
            
            
        img1 = preprocess(sample['image_0']).unsqueeze(0).to(device)
        img2 = preprocess(sample['image_1']).unsqueeze(0).to(device)

        cap1 = clip.tokenize(['a photo of ' + sample['caption_0']]).to(device)        
        cap2 = clip.tokenize(['a photo of ' + sample['caption_1']]).to(device)        

        imgs = torch.cat((img1, img2), dim=0)
        caps = torch.cat((cap1, cap2), dim=0)
        
        with torch.no_grad():
            
            # img score:  2 img, 1 text
            _, logits_per_text1 = model(imgs, cap1) # t1.i1 | t1.i2
            _, logits_per_text2 = model(imgs, cap2) # t1.i1 | t1.i2            
            
            probs_text1 = logits_per_text1.softmax(dim=-1).cpu().numpy() 
            probs_text2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_pred = probs_text1[0][0] # what imgage did the text predict 
            img2_pred = probs_text2[0][1] # t2 pred for i2

            img_scores.append([int(img1_pred > 0.5), int(img2_pred>0.5)])

            #############################################################################
            
            # text score:  1 img, 2 text
            _, logits_per_img1 = model(img1, caps)
            _, logits_per_img2 = model(img2, caps)

            probs_img1 = logits_per_img1.softmax(dim=0).cpu().numpy() 
            probs_img2 = logits_per_img2.softmax(dim=0).cpu().numpy()

            text1_pred = probs_img1[0] # what text did the image predict
            text2_pred = probs_img2[1]

            text_scores.append([int(text1_pred > 0.5), int(text2_pred>0.5)]) 

    out['wino_text_rand_25'] = (np.array([1 for score in text_scores if 0 not in score]).sum())/len(text_scores)
    out['wino_text_rand_50'] = np.array([j for i in text_scores for j in i]).mean()
    out['wino_img_rand_25'] = (np.array([1 for score in img_scores if 0 not in score]).sum())/len(text_scores)
    out['wino_img_rand_50'] = np.array([j for i in img_scores for j in i]).mean()


    return out


    ##################################################################
    #         qid1, text1 , _ = row.split(".")
    #         # Get next row for the pair
    #         row = next(f)
    #         qid2, text2 , _ = row.split(".")
            
    #         qid1, qid2 = int(qid1), int(qid2)
            
    #         img = Image.open(os.path.join(image_dir, f'{qid1}.jpg')).convert("RGB")
    #         # img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg')).convert("RGB")
        
    #         img = clipcap_transform(img)
    #         # img2 = clipcap_transform(img2)

    #         # text1 = 'a photo of ' + statement1
    #         # text2 = 'a photo of ' + statement2

    #         t1_tokens = torch.tensor(gpt2_tokenizer.encode(text1),dtype=torch.int)
    #         t2_tokens = torch.tensor(gpt2_tokenizer.encode(text2),dtype=torch.int)
            
    #         i1_p1, i1_p2 = get_probs(model, img, [t1_tokens, t2_tokens]) #p1>p2
    #         # i2_p1, i2_p2 = get_probs(model, img2, [t1_tokens, t2_tokens]) #p2>p1
            
    #         if c%2==0:
    #             score.append(int(i1_p1 > i1_p2))
    #         else:
    #             score.append(int(i1_p1 < i1_p2))
            

    #         # current_category = categories[num_pairs // 15]
    #         # if rand_25 :
    #         #     if i1_p1.item() > i1_p2.item() and  i2_p1.item() < i2_p2.item():
    #         #         pair_accuracies[current_category] += 1
    #         #         num_pairs += 1

    #         # else: 
    #         #     if i1_p1.item() > i1_p2.item():
    #         #         pair_accuracies[current_category] += 1
                
    #         #     if  i2_p1.item() < i2_p2.item():
    #         #         pair_accuracies[current_category] += 1
                
    #         #     num_pairs += 2

    #         c+=1
    #         #########################################################################################################
    #         # text1 = clip.tokenize([text1]).to(device)
    #         # text2 = clip.tokenize([text2]).to(device)
            
    #         # img1 = preprocess(img1).unsqueeze(0).to(device)
    #         # img2 = preprocess(img2).unsqueeze(0).to(device)
    #         # imgs = torch.cat((img1, img2), dim=0)


    #         # with torch.no_grad():
    #         #     logits_per_image1, logits_per_text1 = model(imgs, text1)
    #         #     logits_per_image2, logits_per_text2 = model(imgs, text2)
                
    #         #     probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
    #         #     probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

    #         # img1_score1 = probs1[0][0]
    #         # img1_score2 = probs2[0][0]
            
    #         # pred1 = "img1" if img1_score1 > 0.5 else "img2"
    #         # pred2 = "img1" if img1_score2 > 0.5 else "img2"

    #         # gt1 = "img1" if qid1 % 2 == 1 else "img2"
    #         # gt2 = "img1" if qid2 % 2 == 1 else "img2"

            
    #         # csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
    #         # current_category = categories[num_pairs // 15]
    #         # if pred1 == gt1 and pred2 == gt2:
    #         #     pair_accuracies[current_category] += 1
    #         # num_pairs += 1

    #     # csv_outfile.close()

    # # Calculate percentage accuracies
    # # for category in pair_accuracies:
    # #     pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    # if rand_25:
    #     out_score = np.array(score).mean()
    # else: 
    #     pairs = []
    #     for i in range(0, len(score), 2):
    #         if i+1 < len(score):
    #             pairs.append((score[i], score[i+1]))

    #     out_score = (np.array([1 for pair in pairs if 0 not in pair]).sum())/len(pairs)

    # return pair_accuracies


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
