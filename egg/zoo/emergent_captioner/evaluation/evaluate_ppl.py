from argparse import ArgumentParser
import torch
from tqdm import tqdm
import more_itertools
from transformers import GPT2Tokenizer, OPTForCausalLM
from collections import namedtuple


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


def pack_sequences(tokenizer: GPT2Tokenizer, max_length, iterable):
    max_length = max_length - tokenizer.num_special_tokens_to_add()
    buffer = []
    for seq in iterable:
        seq = " " + seq
        buffer += tokenizer.encode(seq, add_special_tokens=False)
        while len(buffer) >= max_length:
            tokens = tokenizer.build_inputs_with_special_tokens(buffer[:max_length])
            yield tokens
            buffer = buffer[max_length:]
    if buffer:
        yield tokenizer.build_inputs_with_special_tokens(buffer[:max_length])


def prepare_sequences(tokenizer: GPT2Tokenizer, iterable):
    for seq in iterable:
        seq = " " + seq
        yield tokenizer.encode(seq, add_special_tokens=True)


def make_batches(tokenizer: GPT2Tokenizer, batch_size, iterable):
    for sequences in more_itertools.chunked(iterable, batch_size):
        sequences = list(sequences)
        max_length = max([len(seq) for seq in sequences])
        sequences = [
            seq
            if len(seq) == max_length
            else seq + [tokenizer.pad_token_id] * (max_length - len(seq))
            for seq in sequences
        ]
        input_ids = torch.tensor(sequences)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        yield input_ids, attention_mask


Batch = namedtuple("Batch", ["input_ids", "attention_mask"])


@torch.no_grad()
def ppl_scores(model, iterable):
    device = next(model.parameters()).device
    for input_ids, attention_mask in iterable:
        batch = Batch(input_ids.to(device), attention_mask.to(device))
        input_ids = batch.input_ids[:, :-1]
        attention_mask = batch.attention_mask[:, :-1]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        nlls = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.input_ids[:, 1:].reshape(-1),
            reduction="none",
        )
        nlls = nlls.view(logits.size(0), logits.size(1))
        nlls = nlls[batch.attention_mask[:, :-1].bool()]
        nll = nlls.mean()
        yield nll, attention_mask.long().sum()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("dataset_path")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
    )
    args = parser.parse_args()

    model = OPTForCausalLM.from_pretrained("facebook/opt-350m").eval().to(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

    it = read_plaintext_files(args.dataset_path)
    it = pack_sequences(tokenizer, 512, it)
    it = make_batches(tokenizer, args.batch_size, it)
    it = ppl_scores(model, it)

    tot_tokens = 0
    tot_nll = 0.0
    for nll, n_tokens in it:
        tot_nll = nll * n_tokens + tot_nll
        tot_tokens += n_tokens

    tot_nll /= tot_tokens

    print("Avg PPL:", tot_nll.exp())
