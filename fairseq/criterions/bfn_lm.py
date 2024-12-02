# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
from fairseq import utils
from ._cross_entropy import cross_entropy
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch.functional import F
import numpy as np


# def cross_entropy(logits, target, ignore_index=None, reduction="mean"):
#     lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
#     return F.nll_loss(
#         lprobs,
#         target,
#         ignore_index=ignore_index,
#         reduction=reduction,
#     )


@dataclass
class MaskedLmConfig(FairseqDataclass):

    tpu: bool = II("common.tpu")
    beta1: float = field(
        default=2.0,
        metadata={"help": "beta1 hpram for bayesian flow network"},
    )
    beta_time_order: float = field(
        default=2.0,
        metadata={"help": "time_order for beta hpram for bayesian flow network"},
    )
    loss_type: str = field(
        default="bfl2", metadata={"help": "loss type, choose from bfl2, l2, ce"}
    )
    bf_type: str = field(
        default="mbcltbf",
        metadata={
            "help": "bayesianflow type, choose from mbcltbf, mnbfc1, mnbfc10, mnbfc100"
        },
    )
    sample_steps: int = field(default=100, metadata={"help": "sampling steps for bfn"})

    diff_accuracy: str = field(
        default="dts",
        metadata={"help": "different accuracy type, choose from dts, dtps, dbps"},
    )


def discreteBayesianFlow(t, x, beta1, beta_time_order):
    """
    Args:
        t: [B, N]
        x: [B, N, K], already one-hot
        beta1: [B, N]
    """
    K = x.size(-1)
    beta = beta1 * (t**beta_time_order)  # (B, N)
    beta = beta.unsqueeze(-1)  # (B, N, 1)
    mean = beta * (K * x - 1)  # (B, N, K)
    std = (beta * K).sqrt()  # (B, N, 1)
    eps = torch.randn_like(mean)  # (B, N, K)
    y = mean + std * eps  # (B, N, K)
    theta = F.softmax(y, dim=-1)  # (B, N, K)
    return theta


def discreteBayesianFlow_multinomial(t, x, beta1, beta_time_order=2, steps=100, c=10):
    """
    Args:
        t: [B, N]
        x: [B, N, K], already one-hot
        beta1: [B, N]
    """

    k = x.size(-1)
    torder = beta_time_order
    omega = torch.exp(
        (torch.log(beta1) - np.log(c) - torder * np.log(steps)) / 2
    ).unsqueeze(
        -1
    )  # [B, N, 1]
    lnxi = torch.log(1 + k * omega / (1 - omega))  # [B, N, 1]
    probs_array = x * omega + (1 - omega) / k  # [B, N, K]
    vectorized_func = np.vectorize(
        lambda _t, vector: np.random.multinomial(c * _t**torder, vector),
        signature="(), (n)->(n)",
    )
    # counts = vectorized_func(t.unsqueeze(1).cpu().numpy(), probs_array)
    counts = vectorized_func(t.cpu().numpy(), probs_array.cpu().numpy())
    counts = torch.tensor(counts, dtype=torch.float32).to(t.dtype).to(x.device)
    bflow_probs = torch.softmax(counts * lnxi, dim=-1)
    return bflow_probs


@register_criterion("bfn_lm", dataclass=MaskedLmConfig)
class BFNLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: MaskedLmConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.beta1 = cfg.beta1
        self.beta_time_order = cfg.beta_time_order
        self.num_embeddings = len(task.source_dictionary)
        self.dtype = torch.float16
        self.sample_steps = cfg.sample_steps
        self.bf_type = cfg.bf_type
        self.diff_accuracy = cfg.diff_accuracy
        self.timescheduler = torch.distributions.uniform.Uniform(0, 1)
        self.loss_type = cfg.loss_type

    def discreteCtimeLoss(self, t, x, p_0):
        """Args:
        t: [B, N]
        x: [B, N, K], already one-hot
        p_0: [B, N, K], predicted logits
        """
        # K = x.size(-1)
        K = self.num_embeddings
        coeff = (K * self.beta1 * (self.beta_time_order / 2)) ** 0.5
        L_infinity = t ** (self.beta_time_order - 1) * (
            (coeff * x - coeff * p_0) ** 2
        ).sum(
            dim=-1
        )  # (B, N)
        return L_infinity

    def l2Loss(self, x, p_0):
        """Args:
        x: [B, N, K], already one-hot
        p_0: [B, N, K], predicted logits
        """
        return ((x - p_0) ** 2).sum(dim=-1)  # [B, N]

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # sample["src_tokens"] = sample["net_input"]["src_tokens"] # [B, T]
        _net_input = sample["net_input"]
        sample_size = _net_input["src_lengths"].sum()
        net_input = {k: v for k, v in _net_input.items()}
        target_tokens = sample["target"]
        B, T = target_tokens.size()
        dtype = model.encoder.sentence_encoder.embed_tokens.weight.dtype
        tgt_onehot = F.one_hot(
            target_tokens, num_classes=self.num_embeddings  # [B, T, K]
        ).to(dtype)

        if self.bf_type.startswith("mnbfc"):
            net_input["t"] = (net_input["t"] / self.sample_steps).to(dtype)
        if not self.bf_type.startswith("mnbfc") and self.bf_type not in ["mbcltbf"]:
            raise ValueError(f"bf type {self.bf_type} not supported")
        logits = model(**net_input, **kwargs)[0]  # [B, N, K]
        p_0 = F.softmax(logits, dim=-1)  # [B, N, K]

        mask = torch.arange(T, device=target_tokens.device).view(1, -1) < _net_input[
            "src_lengths"
        ].view(
            -1, 1
        )  # [B, T]
        pred_tokens = torch.argmax(logits, dim=-1)
        inp_tokens = torch.argmax(_net_input["src_tokens"], dim=-1)

        # _mask = (
        #     torch.logical_or(inp_tokens != target_tokens, pred_tokens != target_tokens)
        #     & mask
        # )  # supervise only on error tokens
        _mask = mask & (torch.arange(T, device=target_tokens.device).view(1, -1) > 0)
        # ce_loss = cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     target_tokens.view(-1),
        #     reduction="none",
        #     ignore_index=self.padding_idx,
        # )
        # ce_loss = ce_loss.view(B, T)
        # ce_loss = ce_loss * _mask.float()
        # ce_loss = ce_loss.sum()
        sample_size = _mask.sum()
        ce_loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_tokens.view(-1),
            reduction="none",
            ignore_index=self.padding_idx,
        )
        ce_loss = ce_loss.view(B, T)
        ce_loss = ce_loss * _mask.float()
        ce_loss = ce_loss.sum()

        t = net_input["t"]
        if self.loss_type == "bfl2":
            _losses = self.discreteCtimeLoss(t, tgt_onehot, p_0)  # [B, N]
            _losses = _losses * _mask.float()  # [B, N]
            loss = _losses.sum()
        elif self.loss_type == "l2":
            _losses = self.l2Loss(tgt_onehot, p_0)  # [B, N]
            _losses = _losses * _mask.float()  # [B, N]
            loss = _losses.sum()
        elif self.loss_type == "ce":
            loss = 10 * ce_loss
        else:
            raise ValueError(f"loss type {self.loss_type} not supported")

        _accuracy = ((pred_tokens == target_tokens) & _mask).sum(dim=-1).float() / (
            net_input["src_lengths"].float() - 1
        )
        accuracy = _accuracy.sum()
        std = _accuracy.std() * B

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ce_loss": ce_loss if self.tpu else ce_loss.data,
            "pred_accuracy": accuracy if self.tpu else accuracy.data,
            "pred_accuracy_std": std if self.tpu else std.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_sentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        accuracy = sum(log.get("pred_accuracy", 0) for log in logging_outputs)
        std = sum(log.get("pred_accuracy_std", 0) for log in logging_outputs)
        # if self.training:
        #     self.timescheduler.update(accuracy / n_sentences)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "pred_accuracy",
            accuracy / n_sentences,
            round=3,
        )
        metrics.log_scalar(
            "pred_accuracy_std",
            std / n_sentences,
            round=3,
        )
        # metrics.log_scalar("t_mode", self.timescheduler.ratio, round=3)
        # metrics.log_scalar("schedule_steps", self.timescheduler.step, round=3)
        # metrics.log_scalar(
        #     "schedule_sum_alpha_beta", self.timescheduler.sum_alpha_beta, round=3
        # )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["ce_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
