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

def check_align(a3m_path):
    with open(a3m_path, "r") as f:
        lines = f.readlines()
    collect = []
    for l in lines:
        if l.startswith(">"):
            continue
        collect.append(l.strip())
    collect = [re.sub(r"[a-z]", "", l) for l in collect]

    if( len(set([len(l) for l in collect])) == 1):
        return 
    else:
        print("Find Length Inconsistency, start Aligning sequences: \n")
        return_code = os.system(f"python tools/align_aa_sequences.py -f {a3m_path} -o {a3m_path}")
        print(f"Alignment Process Done")
        return

def profile_from_lists(lsts, dictionary):
    collect = []
    for lst in lsts:
        out = [
            dictionary.index(sym)
            for sym in lst
            if sym in [dictionary.bos_word] + dictionary.symbols[4:24]
        ]
        counts = [0] * len(dictionary.symbols)
        for i in out:
            counts[i] += 1
        collect.append(counts)

    profiles = np.array(collect) + 1e-3
    return profiles / np.sum(profiles, axis=-1, keepdims=True)


def build_profile_from_a3m(a3m_path, dic, prepend_bos=True):
    with open(a3m_path, "r") as f:
        lines = f.readlines()
    collect = []
    for l in lines:
        if l.startswith(">"):
            continue
        collect.append(l.strip())
    collect = [re.sub(r"[a-z]", "", l) for l in collect]
    prepend = [[dic.bos_word] * len(collect)] if prepend_bos else []
    return profile_from_lists(prepend + list(zip(*collect)), dic)


def write_fasta(output_path, sequences):
    with open(output_path, "w") as f:
        for i, sequence in enumerate(sequences):
            f.write(f">seq#{i} L={len(sequence)}\n{sequence}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-a3m", type=str, required=True)
    parser.add_argument("--output-a3m", type=str, required=True)
    args = parser.parse_args()
    model_data = torch.load(args.ckpt_path, map_location="cpu")
    dict_path = f'{glob.glob(model_data["cfg"]["task"]["data"])[0]}/dict.txt'
    dictionary = Dictionary.load(dict_path)
    dictionary.add_symbol("<null_1>")
    dictionary.add_symbol("<mask>")

    cfg = model_data["cfg"]["model"]
    state_dict = upgrade_state_dict(model_data["model"])
    encoder = BFNRobertaEncoder(args=cfg, dictionary=dictionary).cuda()
    assert sorted(encoder.state_dict().keys()) == sorted(state_dict.keys())
    encoder.load_state_dict(state_dict)
    encoder.eval()

    check_align(args.input_a3m)

    input_seq = build_profile_from_a3m(args.input_a3m, dictionary, prepend_bos=True)



    if args.batch_size > 1 and args.num_seqs > args.batch_size:
        b_sz = args.batch_size
        probs = torch.tensor(
            np.repeat(input_seq[np.newaxis], b_sz, axis=0),
            dtype=torch.float32,
            device="cuda",
        )
    else:
        b_sz = 1
        probs = torch.tensor(
            np.expand_dims(input_seq, 0), dtype=torch.float32, device="cuda"
        )

    B, T, D = probs.size()
    src_lengths = torch.tensor([T], dtype=torch.long, device=probs.device)
    output_sequences = []
    tensor_collector = []
    for _i in range((args.num_seqs - 1 + b_sz) // b_sz):
        print(f"Generating batch {_i + 1}/{(args.num_seqs - 1 + b_sz) // b_sz}\n")
        sys.stdout.flush()
        loop_probs = probs
        with torch.no_grad():
            for _t in np.linspace(args.time / 100, 1.0, 500):
                # print(loop_probs)
                t = (
                    torch.ones([B, T], dtype=torch.float32, device=loop_probs.device)
                    * _t
                )
                beta1 = torch.ones_like(t) * model_data["cfg"]["model"].beta1
                loop_probs[:, 0] = (
                    F.one_hot(torch.tensor(dictionary.bos_index), len(dictionary))
                    .float()
                    .to(loop_probs.device)
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
    output_sequences = list(set(output_sequences))
    write_fasta(args.output_a3m, output_sequences)
