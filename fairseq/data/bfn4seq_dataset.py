# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.special import softmax
import torch
from torch.functional import F
from . import BaseWrapperDataset
from fairseq import utils
from fairseq.data import data_utils


class TimeDataset(BaseWrapperDataset):

    def __init__(
        self,
        dataset,
        discrete_time=False,
        dtps_ratio=0.9,
        discrete_sample_steps=100,
        seed=0,
    ):
        """ARGS:
        dataset: dataset to wrap
        discrete_time: whether to use discrete time
        diff_time_per_position: whether to use different time per position
        discrete_sample_steps: number of discrete steps only valid if discrete_time is True
        """
        super().__init__(dataset)
        self.discrete_time = discrete_time
        self.discrete_sample_steps = discrete_sample_steps
        self.dtps_ratio = dtps_ratio
        self.seed = seed

    def _tensor_handler(self, item):
        if self.discrete_time:
            if np.random.rand() < self.dtps_ratio:
                t = torch.randint(1, self.discrete_sample_steps + 1, size=item.shape)
            else:
                t = torch.randint(1, self.discrete_sample_steps + 1, size=[1])
                t = t.repeat(item.shape)
        else:
            if np.random.rand() < self.dtps_ratio:
                t = torch.rand(item.shape)
            else:
                t = torch.rand(1)
                t = t.repeat(item.shape)
        return t

    def _np_handler(self, item):
        if self.discrete_time:
            if np.random.rand() < self.dtps_ratio:
                t = np.random.randint(
                    1, self.discrete_sample_steps + 1, size=item.shape
                )
            else:
                t = np.random.randint(1, self.discrete_sample_steps + 1, size=[1])
                t = np.repeat(t, item.shape)
        else:
            if np.random.rand() < self.dtps_ratio:
                t = np.random.rand(*item.shape)
            else:
                t = np.random.rand(1)
                t = np.repeat(t, item.shape)
        return t

    def __getitem__(self, index):
        item = self.dataset[index]
        with data_utils.numpy_seed(self.seed, self.epoch, index, "TimeDataset"):
            if torch.is_tensor(item):
                return torch.tensor(
                    self._np_handler(item.cpu().numpy()), dtype=torch.float32
                )
            else:
                return self._np_handler(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        raise NotImplementedError


class Beta1Dataset(BaseWrapperDataset):

    def __init__(
        self,
        dataset,
        beta1,
        diff_beta1=False,
        start_beta1=0.2,
        seed=0,
    ):
        """ARGS:
        dataset: dataset to wrap
        diff_beta1: whether to use different beta1 per position
        beta1: starting beta1 value to use, only valid if diff_beta1 is True
        """
        super().__init__(dataset)
        self.beta1 = beta1
        self.diff_beta1 = diff_beta1
        self.start_beta1 = start_beta1
        self.seed = seed

    def _tensor_handler(self, item):
        if self.diff_beta1:
            beta1 = torch.rand(item.shape) * self.beta1 + self.start_beta1
        else:
            beta1 = torch.ones_like(item) * self.beta1
        return beta1

    def _np_handler(self, item):
        if self.diff_beta1:
            beta1 = np.random.rand(*item.shape) * self.beta1 + self.start_beta1
        else:
            beta1 = np.ones_like(item) * self.beta1
        return beta1

    def __getitem__(self, index):
        item = self.dataset[index]
        with data_utils.numpy_seed(self.seed, self.epoch, index, "Beta1Dataset"):
            if torch.is_tensor(item):
                return torch.tensor(
                    self._np_handler(item.cpu().numpy()), dtype=torch.float32
                )
            else:
                return self._np_handler(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        raise NotImplementedError


def pad_vector_sequences(
    values,
    pad_value,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 2d tensors into a padded 3d tensor. pad vectors of shape [T, K] to [B, N, K]."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    K = values[0].size(1)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size, K).fill_(pad_value)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v, res[i][size - len(v) :, ...] if left_pad else res[i][: len(v), ...]
        )
    return res


def _tensor_discreteBayesianFlow_mbcltbf(time, tokens, beta1, dict_size, torder):
    """
    Args:
        time: [..., T]
        tokens: [..., T], to be one-hot encoded
        beta1: [..., T]
        mask: [K], to identify valid amminoacids
    """
    mask = torch.tensor(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=torch.bool,
        device=beta1.device,
    )
    valid_k = 20
    notmask = ~mask
    x = F.one_hot(tokens, num_classes=dict_size).to(beta1.dtype)  # [T, K]
    xmask = x > 0  # [T, K]
    except_mask = (xmask & notmask).any(dim=-1, keepdims=True)  # [T, 1]
    beta = beta1 * (time**torder)  # [T]
    beta = beta.unsqueeze(-1)  # (T, 1)
    _mean = beta * (
        valid_k * x - 1
    )  # (T, K) #TODO fix it by setting irrelevant values to -inf
    mean_mask = torch.where(except_mask, xmask, mask)
    mean = torch.where(mean_mask, _mean, -float("inf"))
    std = (beta * valid_k).sqrt()  # (T, 1)
    eps = torch.randn_like(mean)  # (T, K)
    y = mean + std * eps  # (T, K)

    theta = F.softmax(y, dim=-1)  # (T, K)
    return theta


def sampling_tensor_discreteBayesianFlow_mbcltbf(
    time, tokens, beta1, dict_size, torder
):
    """
    Args:
        time: [..., T]
        tokens: [..., T, K], simplex already
        beta1: [..., T]
        mask: [K], to identify valid amminoacids
    """
    mask = torch.tensor(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=torch.bool,
        device=beta1.device,
    )

    valid_k = 20
    x = tokens
    beta = beta1 * (time**torder)  # [T]
    beta = beta.unsqueeze(-1)  # (T, 1)
    _mean = beta * (
        valid_k * x - 1
    )  # (T, K) #TODO fix it by setting irrelevant values to -inf
    mean = torch.where(mask, _mean, -float("inf"))
    std = (beta * valid_k).sqrt()  # (T, 1)
    eps = torch.randn_like(mean)  # (T, K)
    y = mean + std * eps  # (T, K)
    theta = F.softmax(y, dim=-1)  #  profile as prior   [  y + log(profile) ]
    return theta


def _np_discreteBayesianFlow_mbcltbf(time, tokens, beta1, dict_size, torder):
    """
    Args:
        time: [..., T]
        tokens: [..., T], to be one-hot encoded
        beta1: [..., T]
    """
    mask = np.array(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=np.bool_,
    )
    valid_k = 20
    notmask = ~mask
    x = np.eye(dict_size)[tokens]  # (T, K)
    xmask = x > 0  # (T, K)
    except_mask = (xmask & notmask).any(axis=-1, keepdims=True)  # (T, 1)
    beta = beta1 * (time**torder)  # [T]
    beta = beta[..., np.newaxis]  # (T, 1)
    _mean = beta * (valid_k * x - 1)  # (T, K)
    mean_mask = np.where(except_mask, xmask, mask)
    mean = np.where(mean_mask, _mean, -np.inf)
    std = (beta * valid_k) ** 0.5  # (T, 1)
    eps = np.random.randn(*mean.shape)  # (T, K)
    y = mean + std * eps  # (T, K)
    theta = softmax(y, axis=-1)
    return theta


def _np_discreteBayesianFlow_mnbf(
    time, tokens, beta1, dict_size, torder, steps=100, c=10
):
    """
    Args:
        time: [..., T]
        tokens: [..., T]
        beta1: [.., T]
    """

    mask = np.array(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=np.bool_,
    )
    valid_k = 20
    notmask = ~mask
    x = np.eye(dict_size)[tokens]  # [T, K]
    xmask = x > 0  # [T, K]
    except_mask = (xmask & notmask).any(axis=-1, keepdims=True)  # [T, 1]
    omega = np.exp((np.log(beta1) - np.log(c) - torder * np.log(steps)) / 2)[
        ..., np.newaxis
    ]  # [T, 1]
    lnxi = np.log(1 + valid_k * omega / (1 - omega))  # [T, 1]
    _probs_array = (
        x * omega + (1 - omega) / valid_k
    )  # [T, K] #TODO fix it by setting irrelevant values to 0
    mean_mask = np.where(except_mask, xmask, mask)
    _probs_array = np.where(mean_mask, _probs_array, 0)
    probs_array = _probs_array / _probs_array.sum(axis=-1, keepdims=True)
    vectorized_func = np.vectorize(
        lambda _t, vector: np.random.multinomial(c * _t**torder, vector),
        signature="(), (n)->(n)",
    )
    counts = vectorized_func(time, probs_array)  # [T, K]
    logits = np.where(mean_mask, counts * lnxi, -np.inf)
    bflow_probs = softmax(logits, axis=-1)  # [T, K]
    return bflow_probs


class DiscreteBayesianFlowDataset(BaseWrapperDataset):

    def __init__(
        self,
        dataset,
        time_dataset,
        beta1_dataset,
        dict_size,
        torder=2,
        mode="mbcltbf",
        steps=100,
        c=10,
    ):
        """ARGS:
        dataset: dataset to wrap; shape [T]
        time_dataset: time dataset; shape [T]
        beta1_dataset: beta1 dataset; shape [T]
        dict_size: size of the dictionary
        torder: time order
        mode: mode of the dataset choices are "mbcltbf" or "mnbfc[0-9]*"
        steps: number of steps, valid only if mode is "mnbfc"
        c: constant, valid only if mode is "mnbfc"
        """
        super().__init__(dataset)
        self.time_dataset = time_dataset
        self.beta1_dataset = beta1_dataset
        self.dict_size = dict_size
        self.torder = torder
        self.steps = steps
        self.c = c
        self.mode = mode

    def _tensor_handler(self, time, tokens, beta1):
        if self.mode == "mbcltbf":
            return _tensor_discreteBayesianFlow_mbcltbf(
                time, tokens, beta1, self.dict_size, self.torder
            )
        elif self.mode.startswith("mnbfc"):
            return torch.tensor(
                _np_discreteBayesianFlow_mnbf(
                    time.numpy(),
                    tokens.numpy(),
                    beta1.numpy(),
                    self.dict_size,
                    self.torder,
                    self.steps,
                    self.c,
                ),
                dtype=beta1.dtype,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _np_handler(self, time, tokens, beta1):
        if self.mode == "mbcltbf":
            return _np_discreteBayesianFlow_mbcltbf(
                time, tokens, beta1, self.dict_size, self.torder
            )
        elif self.mode.startswith("mnbfc"):
            return _np_discreteBayesianFlow_mnbf(
                time, tokens, beta1, self.dict_size, self.torder, self.steps, self.c
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
        if hasattr(self.time_dataset, "set_epoch"):
            self.time_dataset.set_epoch(epoch)
        if hasattr(self.beta1_dataset, "set_epoch"):
            self.beta1_dataset.set_epoch(epoch)

    def __getitem__(self, index):
        tokens = self.dataset[index]
        time = self.time_dataset[index]
        beta1 = self.beta1_dataset[index]
        if torch.is_tensor(tokens) and torch.is_tensor(time) and torch.is_tensor(beta1):
            return self._tensor_handler(time, tokens, beta1)
        elif (
            not torch.is_tensor(tokens)
            and not torch.is_tensor(time)
            and not torch.is_tensor(beta1)
        ):
            return self._np_handler(time, tokens, beta1)
        else:
            raise ValueError(
                f"Inconsistent types: {type(tokens)}, {type(time)}, {type(beta1)}"
            )

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return pad_vector_sequences(samples, 0.0, left_pad=False)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return (
            self.dataset.can_reuse_epoch_itr_across_epochs
            and self.time_dataset.can_reuse_epoch_itr_across_epochs
            and self.beta1_dataset.can_reuse_epoch_itr_across_epochs
        )
