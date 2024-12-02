import string
import numpy as np
from typing import List
import tqdm
import argparse
import os
import json

ascii_lowercase_table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
GAP = "-"

def assert_length_eq(
    sequences: List[str],
    primary_sequence: str = None,
    ignore_lower: bool = True,
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
    assert is_eq, (
        f"Length of sequences are equal: {is_len_eq}. "
        f"Sequences should be the same length, "
        f"but got #{np.argmin(is_len_eq)} ({len(sequences[np.argmin(is_len_eq)])})"
        f"compared with primary sequence ({len(primary_sequence)})."
    )
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
    id_matrix = seq_array == np.array(list(primary_sequence.translate(ascii_lowercase_table)))
    non_gaps = seq_array != GAP
    # identity = id_matrix.mean(axis=-1)
    identity = id_matrix.mean(axis=-1).max()
    coverage = non_gaps.mean(axis=-1)
    result = {
        "identity": identity,
        "coverage": coverage,
        "non_gaps": non_gaps,
        "id_matrix": id_matrix,
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None, required=True)
    args = parser.parse_args()

    ref_dir = 'data/MSA_reference'
    a3m_files = [f for f in os.listdir(args.input_dir) if f.endswith('.a3m')]

    file_wise_identities = {}
    for file in a3m_files:
        try:
            with open(os.path.join(args.input_dir, file), 'r') as f:
                filename = file.split('.')[0]
                sequences = f.readlines()
                sequences = [seq.strip() for i,seq in enumerate(sequences) if i%2==1]
                # sequences = list(set(sequences))

                ref_sequences = []
                with open(os.path.join(ref_dir, filename + '.a3m'), 'r') as f:
                    ref_sequences = f.readlines()
                    ref_sequences = [seq.strip() for i,seq in enumerate(ref_sequences) if i%2==1]

                if sequences[0] == ref_sequences[0]:
                    sequences = sequences[1:]

                if sequences[-1] == ref_sequences[0]:
                    sequences = sequences[:-1]

                # if 'profile' in args.input_dir:
                #     sequences = [seq[:-1] for seq in sequences]

                identities = []

                for gen in tqdm.tqdm(sequences):
                    result = calc_id_cov(ref_sequences, gen)
                    identities.append(result['identity'])
                print(f'Average novelty for {filename}: {1 - np.max(identities)}')
                file_wise_identities[filename] = 1 - np.max(identities)
        except Exception as e:
            print(f'\033[31mSome error occured in {file}\033[0m')

    file_wise_identities['all_avg'] = np.mean(list(file_wise_identities.values()))
    with open(f'./results/novelty_{args.input_dir.replace("/", "_")}.json', 'w') as f:
        json.dump(file_wise_identities, f, indent=4)