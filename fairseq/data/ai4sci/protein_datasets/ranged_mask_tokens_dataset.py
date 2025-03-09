# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils

from fairseq.data import BaseWrapperDataset, LRUCacheDataset
from typing import Tuple, List

def trunc_normal_noise(num_mask, noise):
    return np.clip(
            np.random.randn(num_mask, 3) * noise,
            a_min=-noise * 2.0,
            a_max=noise * 2.0,
        )

def normal_noise(num_mask, noise):
    return np.random.randn(num_mask, 3) * noise

def uniform_noise(num_mask, noise):
    return np.random.uniform(
                low=-noise, high=noise, size=(num_mask, 3)
            )

def no_noise(num_mask, noise):
    return 0.0

class RangedMaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        bpe: BPE to use for whole-word masking.
        mask_multiple_length : repeat each mask index multiple times. Default
            value is 1.
        mask_stdev : standard deviation of masks distribution in case of
            multiple masking. Default value is 0.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return LRUCacheDataset(cls(dataset, *args, **kwargs)) 

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        bos_idx:int,
        eos_idx:int,
        aa_vocab_range: Tuple,
        mol_vocab_range: Tuple,
        vocab_special_list: List,
        seq: str,
        aa_mask: str, 
        coords: str, 
        noise_type: str,
        noise: float,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        mask_multiple_length: int = 1,
        mask_stdev: float = 0.0,
        skip_masking: bool = False,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_multiple_length = mask_multiple_length
        self.mask_stdev = mask_stdev
        self.skip_masking = skip_masking
        self.vocab_special_list = vocab_special_list
        self.seq = seq
        self.aa_mask = aa_mask
        self.coords = coords
        self.noise_type = noise_type
        self.noise = noise
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self._sizes = np.array(dataset.sizes)

        if self.bos_idx >= 0:
            self._sizes += 1

        if self.eos_idx >= 0:
            self._sizes += 1

        if random_token_prob > 0.0:
            aa_weights = np.ones(len(self.vocab))
            mol_weights = np.ones(len(self.vocab))
            for special_id in self.vocab_special_list:
                aa_weights[special_id] = 0
                mol_weights[special_id] = 0
            aa_weights[(np.arange(len(self.vocab)) < aa_vocab_range[0]) + (np.arange(len(self.vocab)) > aa_vocab_range[1])] = 0
            mol_weights[(np.arange(len(self.vocab)) < mol_vocab_range[0]) + (np.arange(len(self.vocab)) > mol_vocab_range[1])] = 0
            self.aa_weights = aa_weights / aa_weights.sum()
            self.mol_weights = mol_weights / mol_weights.sum()
 
        self.epoch = 0

        if self.noise_type == "trunc_normal":
            self.noise_f = trunc_normal_noise
        elif self.noise_type == "normal":
            self.noise_f = normal_noise
        elif self.noise_type == "uniform":
            self.noise_f = uniform_noise
        else:
            self.noise_f = no_noise

    @property
    def sizes(self):
        return self._sizes

    # @property
    # def can_reuse_epoch_itr_across_epochs(self):
    #     return True  # only the noise changes, not item sizes

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False  # the item sizes changed

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)
    
    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.bos_idx >= 0:
            n += 1
        if self.eos_idx >= 0:
            n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.bos_idx >= 0:
            n += 1
        if self.eos_idx >= 0:
            n += 1
        return n

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        seed = int(hash((seed, epoch, index)) % 1e6)
        rng = np.random.default_rng(seed)
        # item = self.dataset[index]
        dict_item = self.dataset[index]
        item = dict_item[self.seq]
        aa_mask = dict_item[self.aa_mask]
        aa_mask = np.array(aa_mask, dtype=bool)
        coord = dict_item[self.coords]
        sz = len(item)

        assert (
            self.mask_idx not in item
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            self.mask_idx,
        )
        if self.skip_masking:
            return torch.from_numpy(np.copy(item))

        # decide elements to mask
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * sz / float(self.mask_multiple_length)
            + rng.random()
        )

        # multiple masking as described in the vq-wav2vec paper (https://arxiv.org/abs/1910.05453)
        mask_idc = rng.choice(sz, num_mask, replace=False)
        if self.mask_stdev > 0.0:
            lengths = rng.normal(
                self.mask_multiple_length, self.mask_stdev, size=num_mask
            )
            lengths = [max(0, int(round(x))) for x in lengths]
            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ],
                dtype=np.int64,
            )
        else:
            mask_idc = np.concatenate(
                [mask_idc + i for i in range(self.mask_multiple_length)]
            )
        mask_idc = mask_idc[mask_idc < len(mask)]
        try:
            mask[mask_idc] = True
        except:  # something wrong
            print("Assigning mask indexes {} to mask {} failed!".format(mask_idc, mask))
            raise
        
        targets = np.full(len(mask), self.pad_idx)
        targets[mask] = item[mask]

        # decide unmasking and random replacement
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (rng.random(sz) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = rng.random(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        new_item = np.copy(item)
        new_item[mask] = self.mask_idx
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                aa_rand_mask = rand_mask * aa_mask
                mol_rand_mask = rand_mask * (~aa_mask)
                new_item[aa_rand_mask] = rng.choice(
                    len(self.vocab),
                    aa_rand_mask.sum(),
                    p=self.aa_weights,
                )
                new_item[mol_rand_mask] = rng.choice(
                    len(self.vocab),
                    mol_rand_mask.sum(),
                    p=self.mol_weights,
                )

        mol_mask = mask * (~aa_mask)
        num_mol_mask = mol_mask.astype(np.int32).sum()
        new_coord = np.copy(coord)
        new_coord[mol_mask, :] += self.noise_f(num_mol_mask, self.noise)

        # target_coords = np.copy(coord)
        # target_coords[mol_mask, :] = 0

        new_item = torch.from_numpy(new_item).long()
        new_aa_mask = torch.from_numpy(dict_item[self.aa_mask]).long()
        new_coords = torch.from_numpy(dict_item[self.coords])
        noised_coords = torch.from_numpy(new_coord)
        # target_coords = torch.from_numpy(target_coords)
        target = torch.from_numpy(targets).long()

        if self.bos_idx >= 0:
            new_item = torch.cat([new_item.new([self.bos_idx]), new_item])
            new_aa_mask = torch.cat([new_aa_mask.new([1]), new_aa_mask])
            new_coords = torch.cat([new_coords.new([[0.0, 0.0, 0.0]]), new_coords], dim=0)
            noised_coords = torch.cat([noised_coords.new([[0.0, 0.0, 0.0]]), noised_coords], dim=0)
            # target_coords = torch.cat([target_coords.new([[0.0, 0.0, 0.0]]), target_coords], dim=0)
            target = torch.cat([target.new([self.pad_idx]), target])
        
        if self.eos_idx >= 0:
            new_item = torch.cat([new_item, new_item.new([self.eos_idx])])
            new_aa_mask = torch.cat([new_aa_mask, new_aa_mask.new([1])])
            new_coords = torch.cat([new_coords, new_coords.new([[0.0, 0.0, 0.0]])], dim=0)
            noised_coords = torch.cat([noised_coords, noised_coords.new([[0.0, 0.0, 0.0]])], dim=0)
            # target_coords = torch.cat([target_coords, target_coords.new([[0.0, 0.0, 0.0]])], dim=0)
            target = torch.cat([target, target.new([self.pad_idx])])

        ret = {
            self.seq : new_item,
            self.aa_mask : new_aa_mask.long(),
            self.coords : new_coords,
            "noised_coords" : noised_coords,
            "target":  target,
        }

        return ret
