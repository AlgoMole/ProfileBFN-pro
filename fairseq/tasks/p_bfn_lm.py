# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import II, MISSING, OmegaConf

import torch
from fairseq.optim.amp_optimizer import AMPOptimizer
from glob import glob

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.bfn4seq_dataset import (
    DiscreteBayesianFlowDataset,
    TimeDataset,
    Beta1Dataset,
)
from fairseq.data.bf_tokens_dataset import (
    BayesianFlowTokensDataset as MaskTokensDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


def glob_paths(path_patterns):
    paths = []
    for pattern in path_patterns:
        paths.extend(glob(pattern))
    return list(sorted(paths))


@dataclass
class BFNLMConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )
    include_index: bool = field(
        default=True,
        metadata={"help": "include index in model input. this is used for data2vec"},
    )
    skip_masking: bool = field(
        default=True,
        metadata={"help": "skip masking at dataset"},
    )
    # subsample_train: float = field(
    #     default=1,
    #     metadata={"help": "shorten training set for debugging"},
    # )
    d2v2_multi: bool = field(
        default=False,
        metadata={"help": "prepare dataset for data2vec_multi"},
    )
    n_train_samples: int = field(
        default=-1,
        metadata={
            "help": "how many samples used to train model, only sample once while building dataset first time."
        },
    )
    n_valid_samples: int = field(
        default=-1,
        metadata={
            "help": "how many samples used to validate the model, only sample once while building dataset first time."
        },
    )
    beta1: float = field(
        default=2.0,
        metadata={"help": "beta1 hpram for bayesian flow network"},
    )
    beta_time_order: float = field(
        default=2.0,
        metadata={"help": "time_order for beta hpram for bayesian flow network"},
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
    rebuild_batches: bool = True


@register_task("p_bfn_lm", dataclass=BFNLMConfig)
class ProteinBFNLMTask(FairseqTask):
    cfg: BFNLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: BFNLMConfig, dictionary=None):
        super().__init__(cfg)
        self.dictionary = dictionary or self.load_dict(cfg)

        # add mask token; don't touch this to be compatible with esm
        self.null1_idx = self.dictionary.add_symbol("<null_1>")
        self.mask_idx = self.dictionary.add_symbol("<mask>")
        # TODO mnbfc mbcltbf
        self.torder = self.cfg.beta_time_order
        self.beta1 = self.cfg.beta1
        self.bf_type = self.cfg.bf_type
        self.sample_steps = self.cfg.sample_steps
        self.diff_accuracy = self.cfg.diff_accuracy
        self.c = 1
        if self.cfg.bf_type == "mbcltbf":
            self.discrete_time = False
        elif self.cfg.bf_type.startswith("mnbfc"):
            self.discrete_time = True
            self.c = int(self.cfg.bf_type[5:])
        else:
            raise ValueError(f"bf_type: {self.cfg.bf_type} is not supoorted!")
        self.dtps_ratio = 0.0
        self.diff_beta1 = False

        if self.cfg.diff_accuracy == "dtps":
            self.dtps_ratio = 1.01

        if self.cfg.diff_accuracy.startswith("mtps"):
            self.dtps_ratio = float(self.cfg.diff_accuracy[4:])

        if self.cfg.diff_accuracy == "dbps":
            self.diff_beta1 = True

        if self.cfg.diff_accuracy not in [
            "dts",
            "dtps",
            "dbps",
        ] and not self.cfg.diff_accuracy.startswith("mtps"):
            raise ValueError(
                f"diff_accuracy: {self.cfg.diff_accuracy} is not supoorted!"
            )

    @classmethod
    def setup_task(cls, cfg: BFNLMConfig, **kwargs):
        dictionary = cls.load_dict(cfg)
        return cls(cfg, dictionary)

    @classmethod
    def load_dict(cls, cfg):
        paths = utils.split_paths(cfg.data)
        paths = glob_paths(paths)

        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return dictionary

    def _load_dataset_split(self, split, epoch, combine, n_train_samples=-1):
        paths = utils.split_paths(self.cfg.data)
        paths = glob_paths(paths)

        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # logging.critical(f"loading indexed dataset with: split_path =  {split_path}, combine = {combine}")

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
            n_train_samples=n_train_samples,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )
        
        # logging.critical(f"original fasta dataset [{len(dataset)}] = [{dataset[0].shape, dataset[1].shape, dataset[1].shape}] {dataset[0], dataset[1], dataset[2]}")

        # logging.critical(f"shortening dataset with : split = {split}, shorten_data_split_list = {self.cfg.shorten_data_split_list}, shorten method = {self.cfg.shorten_method}, tokens per sample = {self.cfg.tokens_per_sample}")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,  # internal recombine with epoch and index
            epoch
        )

        # logging.critical(f"shortened fasta dataset [{len(dataset)}] = [{dataset[0].shape, dataset[1].shape, dataset[1].shape}] {dataset[0], dataset[1], dataset[2]}")
        # logging.critical(f"blocking datasets with: size = {dataset.sizes}, block_size = {self.cfg.tokens_per_sample - 1}, break mode = {self.cfg.sample_break_mode}")

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )

        # logging.critical(f"blocked fasta dataset [{len(dataset)}] = [{dataset[0].shape, dataset[1].shape, dataset[1].shape}] {dataset[0], dataset[1], dataset[2]}")

        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        return PrependTokenDataset(dataset, self.source_dictionary.bos())

    # yupei: load our bfn dataset
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split.startswith("train"):
            num_load_samples = self.cfg.n_train_samples
        elif split.startswith("valid"):
            num_load_samples = self.cfg.n_valid_samples
        else:
            num_load_samples = -1
        dataset = self._load_dataset_split(
            split, epoch, combine, n_train_samples=num_load_samples
        )

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )

        # logging.critical(f"dataset before masking {len(dataset)} = {dataset[0], dataset[1], dataset[2]}")

        src_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,  # internal recombine with epoch and index
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
            skip_masking=self.cfg.skip_masking,
            epoch_id=epoch
        )

        # logging.critical(f"dataset after masked {len(dataset)} = {src_dataset[0], src_dataset[1], src_dataset[2]}")

        with data_utils.numpy_seed(
            self.cfg.seed, epoch, hash("load_dataset") % 10**6
        ):  # TODO should involve epoch and index
            shuffle = np.random.permutation(len(src_dataset))

        target_dataset = RightPadDataset(
            src_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        # logging.critical(f"dataset after right padded {len(dataset)} = {src_dataset[0], src_dataset[1], src_dataset[2]}")

        if self.cfg.d2v2_multi:
            dataset = self._d2v2_multi_dataset(src_dataset)
        else:
            dataset = self._regular_dataset(src_dataset, target_dataset)
        self.datasets[split] = SortDataset(
            dataset, sort_order=[shuffle, src_dataset.sizes]
        )

    def _regular_dataset(self, src_dataset, target_dataset):
        time_dataset = TimeDataset(
            src_dataset,
            self.discrete_time,
            self.dtps_ratio,
            self.sample_steps,
            seed=self.cfg.seed,
        )
        beta1_dataset = Beta1Dataset(
            src_dataset,
            self.beta1,
            self.diff_beta1,
            start_beta1=0.2,
            seed=self.cfg.seed,
        )
        input_dict = {
            "src_tokens": DiscreteBayesianFlowDataset(
                src_dataset,
                time_dataset,
                beta1_dataset,
                dict_size=len(self.source_dictionary),
                torder=self.torder,
                mode=self.bf_type,
                steps=self.sample_steps,
                c=self.c,
                use_profile=True
            ),
            "t": RightPadDataset(time_dataset, pad_idx=0),
            "src_lengths": NumelDataset(src_dataset, reduce=False),
        }
        if self.cfg.include_target_tokens:
            input_dict["target_tokens"] = target_dataset
        if self.cfg.include_index:
            input_dict["src_id"] = IdDataset()

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "target": target_dataset,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        )

        return dataset

    def _d2v2_multi_dataset(self, src_dataset):
        input_dict = {
            "source": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "id": IdDataset(),
            "padding_mask": RightPaddingMaskDataset(src_dataset),
        }

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        )
        return dataset

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())

        time_dataset = TimeDataset(
            src_dataset,
            self.discrete_time,
            self.dtps_ratio,
            self.sample_steps,
        )
        beta1_dataset = Beta1Dataset(
            src_dataset, self.beta1, self.diff_beta1, start_beta1=0.2
        )
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": DiscreteBayesianFlowDataset(
                        src_dataset,
                        time_dataset,
                        beta1_dataset,
                        dict_size=len(self.source_dictionary),
                        torder=self.torder,
                        mode=self.bf_type,
                        steps=self.sample_steps,
                        c=self.c,
                        use_profile=True
                    ),
                    "t": RightPadDataset(time_dataset, pad_idx=0),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def begin_epoch(self, epoch, model):
        model.set_epoch(epoch)

    def max_positions(self):
        return self.cfg.tokens_per_sample

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
