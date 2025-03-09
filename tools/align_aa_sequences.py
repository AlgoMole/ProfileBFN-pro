import sys
import argparse

sys.path.append("/home/air/vs_proj/homology-semantic-align")
from tools.general_profile_hmm import ProfileHMM
from absl import logging
from biotite.sequence.io.fasta import FastaFile
from typing import List
from tools import utils
import string
import numpy as np

ascii_lowercase_table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
GAP = "-"


def assert_length_eq(
    sequences: List[str],
    primary_sequence: str = None,
    ignore_lower: bool = False,
    silent: bool = False,
):
    """
    Check if the length of sequences are the same
    """
    if primary_sequence is None:
        primary_sequence = sequences[0]

    if ignore_lower:
        sequences = [s.translate(ascii_lowercase_table) for s in sequences]
        primary_sequence = primary_sequence.translate(ascii_lowercase_table)

    is_len_eq = np.array([len(primary_sequence) == len(seq) for seq in sequences])

    is_eq = is_len_eq.all()
    print(f"Length of sequences are equal: {is_len_eq}")
    err_msg = (
        f"Sequences should be the same length, "
        f"but got #{np.argmin(is_len_eq)} ({len(sequences[np.argmin(is_len_eq)])})"
        f"compared with primary sequence ({len(primary_sequence)})."
    )
    if silent:
        return is_eq, err_msg
    else:
        assert is_eq, err_msg


def calc_id_cov(sequences, primary_sequence: str = None):
    """
    Calculate the identity coverage of two sequences

    Args
    ----------
    sequences: List[str], the list of sequences, the first one is the primary
        sequence. All sequences should be the same length without considering
        lower case letters.

    Returns
    ----------
    result: a dict with keys:
        identity: the identity of the sequences to the primary sequence
        coverage: the coverage of the sequences to the primary sequence
        non_gaps: the non-gaps matrix in the sequences
        id_matrix: the identity matrix of the sequences
    """
    if primary_sequence is None:
        primary_sequence = sequences[0]

    sequences = [s.translate(ascii_lowercase_table) for s in sequences]
    assert_length_eq(sequences, primary_sequence)

    seq_array = np.array([list(seq) for seq in sequences])
    id_matrix = seq_array == np.array(list(primary_sequence))
    non_gaps = seq_array != GAP
    identity = id_matrix.mean(axis=-1)
    coverage = non_gaps.mean(axis=-1)
    result = {
        "identity": identity,
        "coverage": coverage,
        "non_gaps": non_gaps,
        "id_matrix": id_matrix,
    }
    return result


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fasta",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_fasta",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num_iter",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-t",
        "--num_process",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-p",
        "--train_portion",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()
    fasta_file = args.fasta
    output_fasta = args.output_fasta
    num_iter = args.num_iter
    num_process = args.num_process
    train_portion = args.train_portion

    ffile = FastaFile.read(fasta_file)
    key2raw_sequence = {k: list(ffile[k].replace("-", "")) for k in ffile}
    AAtypes = "ARNDCQEGHILKMFPSTWYVX"
    profile_hmm = ProfileHMM(
        match_tokens=AAtypes,
        insert_tokens=AAtypes.lower(),
        gap_token="-",
        insert2match=dict(zip(AAtypes.lower(), AAtypes)),
        psuedo_counts=1.0,
    )
    with utils.timing("align clusters"):
        aligned_sequences = profile_hmm.progressive_msa_fn(
            key2raw_sequence,
            num_process=num_process,
        )
    aakeys = list(ffile.keys())
    wffile = FastaFile(chars_per_line=3000)
    for k in aakeys:
        wffile[k] = "".join(aligned_sequences[k])
    wffile.write(output_fasta)

    primary_sequence = "".join(aligned_sequences[aakeys[0]])
    sequences = ["".join(aligned_sequences[k]) for k in aakeys[1:]]
    id_cov = calc_id_cov(sequences, primary_sequence)
    print(
        f"average identity and coverage are: {id_cov['identity'].mean():.3f}, {id_cov['coverage'].mean():.3f}"
    )
    print(
        "max identity and coverage are: ",
        id_cov["identity"].max(),
        id_cov["coverage"].max(),
    )
    print(
        "min identity and coverage are: ",
        id_cov["identity"].min(),
        id_cov["coverage"].min(),
    )
