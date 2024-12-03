from argparse import ArgumentParser
from tqdm import tqdm
from typing import Iterable

import more_itertools
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_plaintext_files(preds, gold):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """
    with open(preds) as fin1, open(gold) as fin2:
        for p, gg in tqdm(zip(fin1, fin2)):
            p = p.strip()
            gg = gg.strip().split("\t")
            for g in gg:
                yield p, g


def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument("prediction_file", help="File containing predictions.")
    parser.add_argument("gold_file", help="File containing gold captions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def bleurt_scores(model, tokenizer, iterable, batch_size=10):
    device = next(model.parameters()).device
    for batch in more_itertools.chunked(iterable, batch_size):
        pp, gg = zip(*batch)
        batch = tokenizer(pp, gg, padding=True, return_tensors="pt").to(device)
        out1 = model(**batch)[0].squeeze(-1).tolist()
        yield from zip(out1, pp, gg)


def compute_avg_bleurt(
    it: Iterable,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
    model = (
        AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
        .eval()
        .to(device)
    )

    it = bleurt_scores(model, tokenizer, it)

    bleurt1 = {}

    for i, x in enumerate(zip(it)):
        i, o1, p, g = x[0]
        bleurt1[i] = max(bleurt1.get(i, 0.0), o1)
        if verbose:
            print(i, o1, p, g)

    if verbose:
        print("\n\n==========")

    avg_bleurt = sum(bleurt1.values()) / len(bleurt1)
    print(f"BLEURT: {avg_bleurt}")

    return avg_bleurt


def run_evaluation():
    args = get_cli_args()
    it = read_plaintext_files(args.prediction_file, args.gold_file)
    compute_avg_bleurt(it)


if __name__ == "__main__":
    run_evaluation()
