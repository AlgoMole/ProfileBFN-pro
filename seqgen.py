import argparse
import os, sys
import torch
import glob
import re
import numpy as np
import tqdm
import torch.nn.functional as F

from bfn_model.bfn_roberta import BFNRobertaEncoder
from bfn_model.dictionary import Dictionary
from bfn_model.bflow import sampling_tensor_discreteBayesianFlow_mbcltbf


def upgrade_state_dict(state_dict):
    """Removes prefixes 'encoder.sentence_encoder.' and
    'encoder.'"""
    prefixes = ["encoder."]
    pattern = re.compile("^" + "|".join(prefixes))  # ^encoder.sentence_encoder|encoder
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def load_sequence_from_fasta(fasta_path):
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 2
    return lines[1].strip()


def write_fasta(output_path, sequences):
    with open(output_path, "w") as f:
        for i, sequence in enumerate(sequences):
            f.write(f">seq#{i} L={len(sequence)}\n{sequence}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--time", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-fasta", type=str, required=True)
    parser.add_argument("--output-a3m", type=str, required=True)
    args = parser.parse_args()
    model_data = torch.load(args.ckpt_path, map_location="cpu")
    dict_path = f'{glob.glob(model_data["cfg"]["task"]["data"])[0]}/dict.txt'
    dictionary = Dictionary.load(dict_path)
    dictionary.add_symbol("<null_1>")
    dictionary.add_symbol("<mask>")
    # print(len(dictionary.symbols), dictionary.symbols)
    cfg = model_data["cfg"]["model"]
    state_dict = upgrade_state_dict(model_data["model"])
    encoder = BFNRobertaEncoder(args=cfg, dictionary=dictionary).cuda()
    assert sorted(encoder.state_dict().keys()) == sorted(state_dict.keys())
    encoder.load_state_dict(state_dict)
    encoder.eval()
    input_seq = load_sequence_from_fasta(args.input_fasta)

    inputs = torch.cat(
        [
            torch.tensor([dictionary.bos_index], dtype=torch.long),
            dictionary.encode_line(
                input_seq,
                line_tokenizer=list,
                add_if_not_exist=False,
                append_eos=False,
            ),
        ]
    )
    tokens = inputs.unsqueeze(0).cuda()  # [1, T]
    if args.batch_size > 1 and args.num_seqs > args.batch_size:
        b_sz = args.batch_size
        tokens = tokens.repeat(b_sz, 1)
    else:
        b_sz = 1

    B, T = tokens.size()
    src_lengths = torch.tensor([T], dtype=torch.long, device=tokens.device)
    output_sequences = []
    tensor_collector = []
    for _i in range((args.num_seqs - 1 + b_sz) // b_sz):
        print(f"Generating batch {_i + 1}/{(args.num_seqs - 1 + b_sz) // b_sz}\n")
        sys.stdout.flush()
        loop_probs = (
            F.one_hot(tokens, num_classes=len(dictionary)).float().to(tokens.device)
        )
        with torch.no_grad():
            for _t in np.linspace(args.time / 100, 1.0, 500):
                t = torch.ones([B, T], dtype=torch.float32, device=tokens.device) * _t
                beta1 = torch.ones_like(t) * model_data["cfg"]["model"].beta1
                loop_probs[:, 0] = (
                    F.one_hot(torch.tensor(dictionary.bos_index), len(dictionary))
                    .float()
                    .to(tokens.device)
                )
                bfn_pmf = sampling_tensor_discreteBayesianFlow_mbcltbf(
                    t,
                    loop_probs,
                    beta1,
                    dict_size=len(dictionary),
                    torder=model_data["cfg"]["model"].beta_time_order,
                )
                flow_tokens = torch.argmax(bfn_pmf, dim=-1)
                pred_logits, _ = encoder.forward(t, bfn_pmf, src_lengths)
                loop_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
                # print(torch.sum(tokens == flow_tokens) / T)
                # print(torch.sum(tokens == torch.argmax(loop_probs, dim=-1)) / (B * T))
                # print("================")
        tensor_collector.append(torch.argmax(loop_probs, dim=-1).detach().cpu().numpy())

    for tensor in tqdm.tqdm(tensor_collector):
        for i in range(tensor.shape[0]):
            prot = dictionary.string(tensor[i, 1:], separator="")
            output_sequences.append(prot)
    output_sequences = [input_seq] + list(set(output_sequences))
    write_fasta(args.output_a3m, output_sequences)
