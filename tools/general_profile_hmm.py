from collections import Counter
import numpy as np
import re
import copy
from absl import logging
from biotite.sequence.io.fasta import FastaFile
import random
from typing import Any, Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import sys

sys.path.append("/home/air/vs_proj/homology-semantic-align")
from tools import utils

EPSILON = 1e-12


class ProfileHMM:
    def __init__(
        self,
        match_tokens: List,
        insert_tokens: List,
        gap_token: Any,
        insert2match: Dict,
        smooth_prob_matrix: np.ndarray = None,
        psuedo_counts: float = EPSILON,
    ) -> None:
        assert len(match_tokens) == len(insert_tokens)
        self.match_tokens_set = set(list(match_tokens))
        self.match_tokens_list = list(match_tokens)
        self.insert_tokens_set = set(list(insert_tokens))
        self.gap_token = gap_token
        self.insert2match = insert2match
        self.match2insert = {v: k for k, v in insert2match.items()}
        self.match2index = {v: i for i, v in enumerate(match_tokens)}
        self.psuedo_counts = psuedo_counts
        if smooth_prob_matrix is None:
            self.smooth_prob_matrix = np.diag(
                np.ones(len(match_tokens), dtype=np.float32)
            )
        else:
            self.smooth_prob_matrix = smooth_prob_matrix

    def _get_emissions(self, sequences):
        # make emission matrix with psuedo counts
        sequences = np.array(
            [[tok for tok in s if tok not in self.insert_tokens_set] for s in sequences]
        )
        num_seq, seq_len = sequences.shape
        emissions = []

        counts = self.psuedo_counts + np.sum(
            np.expand_dims(sequences, axis=-1)  # (num_seq, seq_len, 1)
            == np.expand_dims(self.match_tokens_list, axis=[0, 1]),  # (1, 1, num_match)
            axis=0,
        )  # (seq_len, num_match)
        counts = np.matmul(counts, self.smooth_prob_matrix)

        # print(np.any(counts == 0))
        emissions = np.log(
            counts
            / (
                np.sum(counts, axis=-1, keepdims=True)
            )  # num_seq + len(self.match_tokens_list)
            * len(self.match_tokens_list)  # with log odds ratio: * num_match
        )  # (seq_len, num_match)
        return emissions

    def _get_insert_emissions(self, sequences):
        num_seq, seq_len = len(sequences), len(sequences[0])
        emissions = np.ones(
            (seq_len + 1, len(self.match_tokens_list)), dtype=np.float32
        )
        for seq in sequences:
            pos = -1
            for r in seq:
                if r in self.match_tokens_set:
                    pos += 1
                elif r in self.insert_tokens_set:
                    emissions[pos, self.match2index[self.insert2match[r]]] += 1
                elif r == self.gap_token:
                    pos += 1
                else:
                    raise ValueError(f"Unexpected character {r}")
        emissions = np.matmul(emissions, self.smooth_prob_matrix)
        emissions = np.log(
            emissions
            / np.sum(emissions, axis=-1, keepdims=True)
            # * len(self.match_tokens_list)
        )
        return emissions

    def _get_transitions(self, sequences):
        num_seq, seq_len = len(sequences), len(sequences[0])
        state_trans = ["M-M", "M-D", "M-I", "D-M", "D-D", "I-M", "I-I"]
        transitions = [
            Counter(state_trans)
            if i < seq_len - 1
            else Counter(
                {"M-M": 0.2, "M-D": 0.2, "M-I": 0.2, "I-I": 0.2, "I-M": 0.2}
            )  # Counter(["M-M", "M-D", "M-I", "I-I", "I-M"])
            for i in range(seq_len)
        ]

        for seq in sequences:
            pos = -1
            for i in range(0, len(seq)):
                if seq[i] in self.match_tokens_set:
                    if i == 0 or seq[i - 1] in self.match_tokens_set:
                        transitions[pos][f"M-M"] += 1
                        pos += 1
                    elif seq[i - 1] in self.insert_tokens_set:
                        transitions[pos][f"I-M"] += 1
                        pos += 1
                    elif seq[i - 1] == self.gap_token:
                        transitions[pos][f"D-M"] += 1
                        pos += 1
                    else:
                        raise ValueError(f"Unexpected character {seq[i-1]}")

                elif seq[i] in self.insert_tokens_set:
                    if i == 0 or seq[i - 1] in self.match_tokens_set:
                        transitions[pos][f"M-I"] += 1
                    elif seq[i - 1] in self.insert_tokens_set:
                        transitions[pos][f"I-I"] += 1
                    elif seq[i - 1] == self.gap_token:
                        logging.warning(f"Gap after Insertion")
                        # transitions[pos][f"D-I"] += 1
                        pass
                    else:
                        raise ValueError(f"Unexpected character {seq[i-1]}")
                elif seq[i] == self.gap_token:
                    if i == 0 or seq[i - 1] in self.match_tokens_set:
                        transitions[pos][f"M-D"] += 1
                        pos += 1
                    elif seq[i - 1] == self.gap_token:
                        transitions[pos][f"D-D"] += 1
                        pos += 1
                    elif seq[i - 1] in self.insert_tokens_set:
                        logging.warning(f"Insertion after Gap")
                        # transitions[pos][f"I-D"] += 1
                        # pos += 1
                        pass
                    else:
                        raise ValueError(f"Unexpected character {seq[i-1]}.")
                else:
                    raise ValueError(f"Unexpected character {seq[i]}.")

        for i, t in enumerate(transitions):
            mtotal = sum([t[k] for k in t.keys() if k[0] == "M"])
            dtotal = sum([t[k] for k in t.keys() if k[0] == "D"])
            itotal = sum([t[k] for k in t.keys() if k[0] == "I"])
            transitions[i] = {
                k: np.log(v / mtotal)
                if k[0] == "M"
                else np.log(v / dtotal)
                if k[0] == "D"
                else np.log(v / itotal)
                for k, v in t.items()
            }
        return transitions

    def _viterbi_decoding(self, emissions, insert_emissions, transitions, sequence):
        e, e_i, t = emissions, insert_emissions, transitions
        # TODO without insert_emissions
        e_i = np.zeros_like(e_i)
        s = [self.match2index[tok] for tok in sequence]
        # insert_emission_cost = np.log(0.05)
        m, n = e.shape[0], len(s)
        dp = np.zeros((m, n, 3))
        trace = np.zeros((m, n, 3), dtype=int)
        dp[0, 0, :] = t[-1]["M-M"], -np.inf, -np.inf
        for i in range(1, m):
            dp[i, 0, 0] = (
                t[-1]["M-D"]
                + t[i - 1]["D-M"]
                + sum([t[_j]["D-D"] for _j in range(i - 1)])
                + e[i, s[0]]
            )
            trace[i, 0, 0] = 1
            _scores = [
                dp[i - 1, 0, 0] + t[i - 1]["M-D"],
                dp[i - 1, 0, 1] + t[i - 1]["D-D"],
                -np.inf,
            ]
            _trace = np.argmax(_scores)
            dp[i, 0, 1] = _scores[_trace]
            trace[i, 0, 1] = _trace
            dp[i, 0, 2] = -np.inf
            trace[i, 0, 2] = -1

        for j in range(1, n):
            dp[0, j, 0] = (
                t[-1]["M-I"]
                + t[-1]["I-M"]
                + sum([t[-1]["I-I"] for _ in range(j - 1)])
                + e_i[-1, s[j]] * j
                + e[0, s[j]]
            )
            trace[0, j, 0] = 2
            dp[0, j, 1] = -np.inf
            trace[0, j, 1] = -1

            _scores = [
                dp[0, j - 1, 0] + t[0]["M-I"],
                -np.inf,
                dp[0, j - 1, 2] + t[0]["I-I"],
            ]
            _trace = np.argmax(_scores)
            dp[0, j, 2] = _scores[_trace] + e_i[0, s[j]]
            trace[0, j, 2] = _trace

        for i in range(1, m):
            for j in range(1, n):
                # match
                _scores = [
                    dp[i - 1, j - 1, 0] + t[i - 1]["M-M"],
                    dp[i - 1, j - 1, 1] + t[i - 1]["D-M"],
                    dp[i - 1, j - 1, 2] + t[i - 1]["I-M"],
                ]
                _trace = np.argmax(_scores)
                dp[i, j, 0] = _scores[_trace] + e[i, s[j]]
                trace[i, j, 0] = _trace
                # delete
                _scores = [
                    dp[i - 1, j, 0] + t[i - 1]["M-D"],
                    dp[i - 1, j, 1] + t[i - 1]["D-D"],
                    dp[i - 1, j, 2] + t[i - 1].get("I-D", -np.inf),
                ]
                _trace = np.argmax(_scores)
                dp[i, j, 1] = _scores[_trace]
                trace[i, j, 1] = _trace

                # insert
                _scores = [
                    dp[i, j - 1, 0] + t[i]["M-I"],
                    dp[i, j - 1, 1] + t[i].get("D-I", -np.inf),
                    dp[i, j - 1, 2] + t[i]["I-I"],
                ]
                _trace = np.argmax(_scores)
                dp[i, j, 2] = _scores[_trace] + e_i[i, s[j]]
                trace[i, j, 2] = _trace

        # print(np.transpose(dp[:4, :4, :], (2, 0, 1)))
        # print(np.transpose(trace[:4, :4, :], (2, 0, 1)))

        return dp, trace

    def _traceback(self, trace, i, j, k):
        # if i < 5 and j < 5:
        #     print(i, j, k)
        try:
            if i < 0 and j < 0:
                return ""
            if j < 0:
                assert k == 1, f"k must be 1 instead of {k}"
                return self._traceback(trace, i - 1, j, k) + "D"
            if i < 0:
                assert k == 2, f"k must be 2 instead of {k}"
                return self._traceback(trace, i, j - 1, k) + "I"
        except:
            print(i, j, k)
            raise

        if k == 0:
            return self._traceback(trace, i - 1, j - 1, trace[i, j, k]) + "M"
        elif k == 1:
            return self._traceback(trace, i - 1, j, trace[i, j, k]) + "D"
        elif k == 2:
            return self._traceback(trace, i, j - 1, trace[i, j, k]) + "I"
        else:
            print(i, j, k)
            raise ValueError(f"Invalid trace {k}")

    def _traceback2(self, trace, init_k):
        states = "MDI"
        _ret = ""
        m, n, _ = trace.shape
        i, j = m - 1, n - 1
        k = init_k
        while i >= 0 or j >= 0:
            next_k = trace[i, j, k]
            _ret = states[k] + _ret
            if j < 0:
                assert k == 1, f"k must be 1 instead of {k}"
                i -= 1
                continue
            if i < 0:
                assert k == 2, f"k must be 2 instead of {k}"
                j -= 1
                continue

            if k == 0:
                i -= 1
                j -= 1
            elif k == 1:
                i -= 1
            elif k == 2:
                j -= 1
            else:
                print(i, j, k)
                raise ValueError(f"Invalid trace {k}")

            k = next_k

        return _ret

    def get_trace(self, seq, states):
        aln = (
            [0] * len(states)
            if isinstance(seq, list)
            else np.zeros(len(states), dtype=seq.dtype)
        )
        i = 0
        for r in seq:
            if states[i] == "D":
                while i < len(states) and states[i] == "D":
                    aln[i] = self.gap_token
                    i += 1
            if states[i] == "M":
                aln[i] = r
                i += 1
            elif states[i] == "I":
                aln[i] = self.match2insert[r]
                i += 1
            else:
                raise ValueError(f"Invalid state {states[i]}")
        while i < len(states) and states[i] == "D":
            aln[i] = self.gap_token
            i += 1
        return aln

    def get_trace2AAstring(self, aaseq, states):
        aln = ""
        i = 0
        for r in aaseq:
            if states[i] == "D":
                while i < len(states) and states[i] == "D":
                    aln += "-"
                    i += 1
            if states[i] == "M":
                aln += r
                i += 1
            elif states[i] == "I":
                aln += r.lower()
                i += 1
            else:
                raise ValueError(f"Invalid state {states[i]}")
        while i < len(states) and states[i] == "D":
            aln += "-"
            i += 1
        return aln

    def _worker(self, k2seq, emissions, emissions_i, transitions):
        k, seq = k2seq
        dp_score, trace = self._viterbi_decoding(
            emissions, emissions_i, transitions, seq
        )
        # states_ = self._traceback(
        #     trace, len(emissions) - 1, len(seq) - 1, np.argmax(dp_score[-1, -1, :])
        # )
        states = self._traceback2(trace, init_k=dp_score[-1, -1, :].argmax())
        _no_gap = re.sub(r"[D]", "", states)
        assert len(_no_gap) == len(seq), f"{len(_no_gap)} != {len(seq)}"
        a3m = self.get_trace(seq, states)
        return k, a3m, dp_score[-1, -1, :].max()

    def msa_fn(self, raw_key2sequence, num_iter=20, num_process=1, train_portion=0.5):
        keys = list(raw_key2sequence.keys())
        aligned_key2sequence = {keys[0]: raw_key2sequence[keys[0]]}

        def _one_step(aligned_seq_keys, raw_seq_keys):
            aligned_sequences = [aligned_key2sequence[k] for k in aligned_seq_keys]
            emissions = self._get_emissions(sequences=aligned_sequences)
            emissions_i = self._get_insert_emissions(sequences=aligned_sequences)
            transitions = self._get_transitions(sequences=aligned_sequences)

            pworker = partial(
                self._worker,
                emissions=emissions,
                emissions_i=emissions_i,
                transitions=transitions,
            )
            with ProcessPoolExecutor(num_process) as executor:
                results = executor.map(
                    pworker,
                    list(
                        zip(raw_seq_keys, [raw_key2sequence[k] for k in raw_seq_keys])
                    ),
                )
            # results = []
            # for k in raw_seq_keys:
            #     seq = raw_key2sequence[k]
            #     results.append(pworker((k, seq)))
            scores = []
            for k, a3m, _scr in results:
                aligned_key2sequence[k] = a3m
                scores.append(_scr)
            return scores

        for i in range(num_iter):
            with utils.timing(f"iteration {i}"):
                selected_ids = np.random.choice(
                    np.arange(1, len(aligned_key2sequence)),
                    int(len(aligned_key2sequence) * train_portion),
                    replace=False,
                )
                aligned_keys = [keys[0]] + [keys[i] for i in selected_ids]
                raw_keys = [
                    keys[i] for i in np.setdiff1d(np.arange(1, len(keys)), selected_ids)
                ]
                scores = _one_step(aligned_keys, raw_keys)
                print(f"iteration {i} score {np.mean(scores)}")
        return aligned_key2sequence

    def progressive_msa_fn(self, raw_key2sequence, num_process=1):
        keys = list(raw_key2sequence.keys())
        aligned_key2sequence = {keys[0]: raw_key2sequence[keys[0]]}

        def _one_step(aligned_seq_keys, raw_seq_keys):
            aligned_sequences = [aligned_key2sequence[k] for k in aligned_seq_keys]
            emissions = self._get_emissions(sequences=aligned_sequences)
            emissions_i = self._get_insert_emissions(sequences=aligned_sequences)
            transitions = self._get_transitions(sequences=aligned_sequences)

            pworker = partial(
                self._worker,
                emissions=emissions,
                emissions_i=emissions_i,
                transitions=transitions,
            )
            with ProcessPoolExecutor(num_process) as executor:
                results = executor.map(
                    pworker,
                    list(
                        zip(raw_seq_keys, [raw_key2sequence[k] for k in raw_seq_keys])
                    ),
                )

            results = sorted(
                [(k, a3m, _scr) for k, a3m, _scr in results],
                key=lambda x: x[2],
                reverse=True,
            )

            scores = []
            for k, a3m, _scr in results[: min(len(aligned_key2sequence), 512)]:
                # if k in aligned_key2sequence and any(
                #     [a != b for a, b in zip(a3m, aligned_key2sequence[k])]
                # ):
                #     aligned_key2sequence[f"{k}_1"] = a3m
                # else:
                aligned_key2sequence[k] = a3m
                scores.append(_scr)
            return scores

        i = 0
        while len(aligned_key2sequence) < len(raw_key2sequence):
            with utils.timing(f"hmm aligning iteration {i}"):
                aligned_keys = list(aligned_key2sequence.keys())
                raw_keys = [k for k in raw_key2sequence.keys() if k not in aligned_keys]
                scores = _one_step(aligned_keys, raw_keys)
                print(
                    f"iteration {i} score {np.mean(scores)}, {len(scores)} sequences aligned to profile hmm"
                )
                i += 1
        with utils.timing(f"final hmm alignment"):
            aligned_keys = list(aligned_key2sequence.keys())
            raw_keys = list(raw_key2sequence.keys())[1:]
            scores = _one_step(aligned_keys, raw_keys)
            print(f"final hmm iteration score {np.mean(scores)}")

        return aligned_key2sequence
