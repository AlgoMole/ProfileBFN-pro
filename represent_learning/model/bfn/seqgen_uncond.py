import argparse
import os, sys
import torch
import glob
import re
import numpy as np
import tqdm

from fairseq.models.roberta.bfn_roberta import BFNRobertaEncoder
from fairseq.models.roberta.p_roberta import ESM2RobertaEncoder
from fairseq.data import Dictionary
from fairseq.data.bfn4seq_dataset import (
    _tensor_discreteBayesianFlow_mbcltbf,
    _np_discreteBayesianFlow_mbcltbf,
    _np_discreteBayesianFlow_mnbf,
)


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
    # with open(os.path.join(output_path, "output"), "w") as f:
    with open(output_path, "w") as f:
        for i, sequence in enumerate(sequences):
            f.write(f">SEQUENCE_{i}_L={len(sequence)}\n{sequence}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start_t", type=float, default=0.5)
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--input-fasta", type=str, required=True)
    parser.add_argument("--output-a3m", type=str, required=True)
    parser.add_argument("--prior", type=str, required=True)
    parser.add_argument('--scaffold-min', type=int, default=50,
                        help="Min scaffold len ")
    parser.add_argument('--scaffold-max', type=int, default=100,
                        help="Max scaffold len, will randomly choose a value between min/max")
    args = parser.parse_args()
    print(args)
    model_data = torch.load(args.ckpt_path, map_location="cpu")
    dict_path = f'{glob.glob(model_data["cfg"]["task"]["data"])[0]}/dict.txt'
    dictionary = Dictionary.load(dict_path)
    dictionary.add_symbol("<null_1>")
    dictionary.add_symbol("<mask>")

    print(len(dictionary.symbols), dictionary.symbols)
    cfg = model_data["cfg"]["model"]
    state_dict = upgrade_state_dict(model_data["model"])
    if(args.model_arch.startswith("bfn")):
        print(f"Using BFN sampling: ")
        encoder = BFNRobertaEncoder(args=cfg, dictionary=dictionary).cuda()
    else:
        encoder = ESM2RobertaEncoder(args=cfg, dictionary=dictionary).cuda()
    assert sorted(encoder.state_dict().keys()) == sorted(state_dict.keys())
    encoder.load_state_dict(state_dict)
    encoder.eval()
    input_seq = load_sequence_from_fasta(args.input_fasta)
    print(f"Input sequence: {input_seq}")

    input_line = dictionary.encode_line(
                input_seq,
                line_tokenizer=list,
                add_if_not_exist=False,
                append_eos=False,
            )

    if(args.prior == "file"):
        pass
    elif(args.prior == "mask"):
        input_line = torch.tensor([dictionary.index("<mask>")] * len(input_line), dtype=torch.long)
    elif(args.prior == "rand"):
        input_line = torch.randint(0, len(dictionary), (len(input_line),), dtype=torch.long)

    inputs = torch.cat(
        [
            torch.tensor([dictionary.bos_index], dtype=torch.long),
            input_line,
        ]
    )
    print(f"Input tokens: {inputs.shape} {inputs}")

    tokens = inputs.unsqueeze(0).cuda()
    if args.batch_size > 1 and args.num_seqs >= args.batch_size:
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
        loop_tokens = tokens
        for _t in np.linspace(args.start_t, 1.0, args.max_iters): # sample steps
            # t = torch.rand([B, T], dtype=torch.float32, device=tokens.device) 
            t = torch.ones([B, T], dtype=torch.float32, device=tokens.device) * _t
            loop_tokens[:, 0] = dictionary.bos_index

            if(args.model_arch.startswith("bfn")):
                beta1 = torch.ones_like(t) * model_data["cfg"]["model"].beta1
                bfn_tokens = _tensor_discreteBayesianFlow_mbcltbf(
                    t,
                    loop_tokens,
                    beta1,
                    dict_size=len(dictionary),
                    torder=model_data["cfg"]["model"].beta_time_order,
                )
                pred_tokens, _ = encoder.forward(t, bfn_tokens, src_lengths)    # [B, T, V]
            else:
                pred_tokens, _ = encoder.forward(loop_tokens)    # [B, T, V]

            pred_tokens = torch.nn.functional.softmax(pred_tokens, dim=-1)  # [B, T, V]
            B, T, V = pred_tokens.size()
            pred_tokens = pred_tokens.view(B * T, V)
            loop_tokens = torch.multinomial(pred_tokens, 1, replacement=True)
            loop_tokens = loop_tokens.view(B, T)

        tensor_collector.append(loop_tokens.detach().cpu().numpy())
    for tensor in tqdm.tqdm(tensor_collector):
        for i in range(tensor.shape[0]):
            prot = dictionary.string(tensor[i, 1:], separator="")
            output_sequences.append(prot)
    write_fasta(args.output_a3m, output_sequences)
