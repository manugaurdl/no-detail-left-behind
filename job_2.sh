source ../venvs/egg/bin/activate

CUDA_VISIBLE_DEVICES=1 python -m egg.zoo.emergent_captioner.finetuning.train sr_medium mistral
CUDA_VISIBLE_DEVICES=1 python -m egg.zoo.emergent_captioner.finetuning.train sr_easy mistral
