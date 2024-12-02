# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn.functional as F
from fairseq import modules, utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MaskedLmConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("masked_lm_with_dist_loss", dataclass=MaskedLmConfig)
class MaskedLmWithDistLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: MaskedLmConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        aa_mask = sample["net_input"]["aa_mask"]
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits, _, encoder_distance, x_norm = model(**sample["net_input"], masked_tokens=masked_tokens, **kwargs)
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        masked_pred = logits.argmax(dim=-1)
        masked_hit = (masked_pred == targets).long().sum()
        masked_cnt = sample_size

        if masked_cnt == 0:
            masked_cnt += 1

        logging_output = {
            "masked_token_loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        if encoder_distance is not None:
            dist_masked_tokens = torch.logical_and(masked_tokens, ~aa_mask)
            masked_dist_loss = self.cal_dist_loss(
                sample, encoder_distance, dist_masked_tokens, normalize=True
            )
            loss = loss * self.args.masked_token_loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        logging_output["loss"] = loss if self.tpu else loss.data
        
        return loss, sample_size, logging_output

    def cal_dist_loss(self, sample, dist, masked_tokens, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=3
        )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 1) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
