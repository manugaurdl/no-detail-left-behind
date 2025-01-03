
# No Detail Left Behind: Revisiting Self-Retrieval for Fine-Grained Image Captioning

[Manu Gaur](https://manugaurdl.github.io/), [Darshan Singh S](https://darshansingh11.github.io/), [Makarand Tapaswi](https://makarandtapaswi.github.io/)

<p align="center">
  <img src="imgs/nldb_teaser.jpg"/>
</p>

<p float="left">
  <a href="https://arxiv.org/abs/2409.03025">
    <img src="https://img.shields.io/badge/arXiv-2409.03025-b31b1b.svg" alt="arXiv"/>
  </a>
  <a href="https://katha-ai.github.io/projects/no-detail-left-behind/">
    <img src="https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white" alt="Github Pages"/>
  </a>
<a href="https://huggingface.co/manu-gaur/NDLB" target="_blank">
    <img alt="HF Model: NDLB" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-NDLB-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/manu-gaur/NDLB-TrueMatch-Benchmark" target="_blank">
    <img alt="HF Dataset: TrueMatch" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-TrueMatch-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/manu-gaur/NDLB_data" target="_blank">
    <img alt="HF Dataset: VCB Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-Visual_Caption_Boosting-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
</p>


## Welcome to the code repository for TMLR 2024 accepted paper No Detail Left Behind. 
### This repository contains data and code for MLE training, REINFORCE fine-tuning, and self-retrieval evaluation on the <i>TrueMatch</i> benchmark.

### :zap:	For instant visualization of data samples, please visit our [Project Page](https://katha-ai.github.io/projects/no-detail-left-behind/)

## Getting Started

### Installation
```
conda create -n d3 python=3.10 -y
conda activate ndlb
pip install -r requirements.txt
```


## Setting-up Data 💿

- The data is available [Here](https://drive.google.com/file/d/1V7TDA16cZML1JhABcJocBLRtAyOYfqWc/view?usp=sharing) as a `.zip` file.
- Either manually visit the link and donwload the `dataset.zip` in the root of this directory, or
- Download the `dataset.zip` via `gdown`

```
pip install gdown
cd no-detail-left-behind
gdown 1V7TDA16cZML1JhABcJocBLRtAyOYfqWc
```

Unzip the contents of `dataset.zip`, and ensure the following directory structure

```
├── images
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── ...
│   ├── ...
└── dataset.json
```

## Self-Retrieval Evaluation on <i>TrueMatch</i>


To evaluate the descriptions (stored as `./captions/gpt4o.json`) on the <i>TrueMatch</i> benchmark, run 
```
python evaluate.py gpt4o
```
The script conducts self-retreival evaluation using captions stored in `./captions/gpt4o.json` and stores an pickled dictionary containing recall scores for each image pair in `./uid2acc/`. The script currently uses `google/siglip-so400m-patch14-384` as a scorer.

## Self-Retrieval Evaluation on <i>TrueMatch</i>


To MLE train CLIPCap on a particular dataset (COCO, BlendCap, HolisticCap), run:
```
python -m egg.zoo.emergent_captioner.finetuning.train dataset mle
```

To REINFORCE fine-tune a model with a particular reward (SR, CIDEr) that has been MLE trained with some dataset, run:
```
python -m egg.zoo.emergent_captioner.finetuning.train dataset reward
```


## BibTeX
If you find our work useful, please cite as below

```
@article{gaur2024detect,
  title={No Detail Left Behind: Revisiting Self-Retrieval for Fine-Grained Image Captioning},
  author={Gaur, Manu and Singh S, Darshan and Tapaswi, Makarand.},
  journal={arXiv preprint arXiv:2409.03025},
  year={2024}
}
```

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


