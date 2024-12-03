from argparse import ArgumentParser
from typing import List, Union

import torch


def extract_gt_from_interaction(interaction, multi_reference: bool = False):
    all_captions = []
    for batch in interaction.aux_input["captions"]:
        if multi_reference:
            for batch_captions in zip(*batch):
                all_captions.append(list(batch_captions))
        else:
            all_captions.extend(batch)

    return all_captions


def write_captions_to_file(
    captions: Union[List[str], List[List[str]]], output_file: str, multi_reference: bool
):
    with open(output_file, "w") as f:
        for reference in captions:
            if multi_reference:
                c = [caption.strip().replace("\n", "") for caption in reference]
            else:
                c = [reference.strip().replace("\n", "")]
            f.write("\t".join(c))
            f.write("\n")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--interaction_file", "-i", required=True)
    parser.add_argument("--output_file", "-o", required=True)
    parser.add_argument("--multi_reference", "-m", action="store_true", default=False)
    return parser.parse_args()


def extract_gt():
    args = get_args()
    interaction = torch.load(args.interaction_file)
    captions = extract_gt_from_interaction(interaction, args.multi_reference)
    write_captions_to_file(captions, args.output_file, args.multi_reference)


if __name__ == "__main__":
    extract_gt()
