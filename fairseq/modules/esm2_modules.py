# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from fairseq.models.transformer import TransformerConfig

# from .rotary_multihead_attention import RotaryMultiheadAttention
# from .axial_attention import ColumnSelfAttention, RowSelfAttention


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias=True,
        bos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.eos_idx is not None:
            eos_num = (torch.sum(tokens.eq(self.eos_idx), dim=1) > 0)
            if eos_num.all(): # remove eos
                eos_mask = tokens.ne(self.eos_idx).to(attentions)
                eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
                attentions = attentions * eos_mask[:, None, None, :, :]
                attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.bos_idx is not None:
            bos_num = (torch.sum(tokens.eq(self.bos_idx), dim=1) > 0)
            if bos_num.all(): # remove bos
                attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))

