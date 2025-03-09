# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import torch
import numpy as np
from fairseq.data import Dictionary
from functools import lru_cache
from fairseq.data import BaseWrapperDataset


class KeyTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key,
        dictionary: Dictionary,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.key=key
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        item = self.dataset[index]
        raw_data = item[self.key]
        assert len(raw_data) < self.max_seq_len and len(raw_data) > 0
        item[self.key] = np.array([self.dictionary.index(x) for x in raw_data], dtype=np.int32)
        return item