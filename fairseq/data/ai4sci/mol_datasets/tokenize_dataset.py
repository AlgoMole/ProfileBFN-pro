# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import torch
from fairseq.data import Dictionary
from functools import lru_cache
from fairseq.data import BaseWrapperDataset
import numpy as np

class TokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        assert len(raw_data) < self.max_seq_len and len(raw_data) > 0
        # return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()
        new_data = np.array([self.dictionary.index(x + '_a') for x in raw_data], dtype=np.int32)
        return torch.from_numpy(new_data)