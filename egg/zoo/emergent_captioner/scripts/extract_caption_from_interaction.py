from argparse import ArgumentParser
from typing import List

import torch


def extract_captions_from_interaction(interaction):
    captions = [
        caption.strip().replace("\n", "")
        for batch_captions in interaction.message
        for caption in batch_captions
    ]
    return captions


def write_captions_to_file(captions: List[List[str]], output_file: str):
    with open(output_file, "w") as f:
        for caption in captions:
            f.write(f"{caption}\n")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--interaction_file", "-i", required=True)
    parser.add_argument("--output_file", "-o", required=True)
    return parser.parse_args()


def extract_captions():
    args = get_args()
    interaction = torch.load(args.interaction_file)
    captions = extract_captions_from_interaction(interaction)
    write_captions_to_file(captions, args.output_file)


if __name__ == "__main__":
    extract_captions()
