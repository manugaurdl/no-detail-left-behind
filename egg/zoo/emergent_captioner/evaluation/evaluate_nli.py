from argparse import ArgumentParser
from typing import Iterable

import more_itertools
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument("prediction_file", help="File containing predictions.")
    parser.add_argument("gold_file", help="File containing gold captions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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


@torch.no_grad()
def entailment_scores(model, tokenizer, iterable, batch_size=10):
    device = next(model.parameters()).device
    for batch in more_itertools.chunked(iterable, batch_size):
        ii, pp, gg = zip(*batch)
        batch = tokenizer(pp, gg, padding=True, return_tensors="pt").to(device)
        out1 = model(**batch).logits.softmax(-1)[:, 1].tolist()
        batch = tokenizer(gg, pp, padding=True, return_tensors="pt").to(device)
        out2 = model(**batch).logits.softmax(-1)[:, 1].tolist()
        yield from zip(ii, out1, out2, pp, gg)


def compute_avg_nli_scores(
    it: Iterable,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = (
        AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        .eval()
        .to(device)
    )

    it = entailment_scores(model, tokenizer, it)

    entailment1 = {}
    entailment2 = {}
    entailment3 = {}

    for x in zip(it):
        i, o1, o2, p, g = x[0]
        entailment1[i] = max(entailment1.get(i, 0.0), o1)
        entailment2[i] = max(entailment2.get(i, 0.0), o2)
        entailment3[i] = max(entailment1[i], entailment2[i])
        if verbose:
            print(i, o1, o2, p, g)

    if verbose:
        print("\n\n==========")

    avg_nli_scores = sum(entailment2.values()) / len(entailment1)

    print("pred -> gold:", avg_nli_scores)

    if verbose:
        print("pred <- gold:", sum(entailment1.values()) / len(entailment2))
        print("max:", sum(entailment3.values()) / len(entailment3))

    return avg_nli_scores


def run_evaluation():
    args = get_cli_args()
    it = read_plaintext_files(args.prediction_file, args.gold_file)
    compute_avg_nli_scores(it)


if __name__ == "__main__":
    run_evaluation()
