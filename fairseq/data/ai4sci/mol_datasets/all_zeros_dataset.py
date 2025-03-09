# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
import logging
from fairseq.data import BaseWrapperDataset
import torch

logger = logging.getLogger(__name__)


class AllZerosDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        return torch.zeros_like(self.dataset[index]).long()
    