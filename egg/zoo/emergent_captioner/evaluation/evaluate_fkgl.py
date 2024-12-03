from argparse import ArgumentParser

from tqdm import tqdm
import textstat


def read_plaintext_files(preds):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """
    with open(preds) as fin1:
        for p in tqdm(fin1):
            pp = p.strip().split("\t")
            for p in pp:
                yield p


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("dataset_path")

    args = parser.parse_args()

    it = read_plaintext_files(args.dataset_path)

    tot_snt = 0
    tot_fkgl = 0.0
    for snt in it:
        tot_fkgl += textstat.flesch_kincaid_grade(snt)
        tot_snt += 1

    print(f"Avg FKGL: {tot_fkgl / tot_snt}")
