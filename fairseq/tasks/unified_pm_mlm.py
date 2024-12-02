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

from fairseq import utils
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)

from fairseq.data.ai4sci import  (
    LMDBDataset,
    Add2DConformerDataset,
    AllZerosDataset,
    KeyDataset,
    ConformerSampleDataset,
    AtomTypeDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    TokenizeDataset,
    MaskPointsDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    EdgeTypeDataset,
    DistanceDataset,
    FromNumpyDataset,
    RightPadDatasetCoord, 
    RightPadDatasetCross2D, 
    RightPadDataset2D,
    ProteinsUnfoldDataset,
    KeyTokenizeDataset,
    RangedMaskTokensDataset,
    ProteinsDistanceDataset,
)

from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


@dataclass
class MaskedLMConfig(FairseqDataclass):
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
    pro_sample_ratio: float = field(
        default=1.0,
        metadata={"help": "the sample ratio of proteins dataset"},
    )
    mol_sample_ratio: float = field(
        default=1.0,
        metadata={"help": "the sample ratio of molecules dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    unfold_prob: float = field(
        default=0.05,
        metadata={"help": "probability of unfolding a AA"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
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
        default=False,
        metadata={"help": "skip masking at dataset"},
    )
    remove_hydrogen: bool = field(
        default=False,
        metadata={"help": "remove hydrogen atoms"},
    )
    remove_polar_hydrogen: bool = field(
        default=False,
        metadata={"help": "remove polar hydrogen atoms"},
    )
    noise_type: str = field(
        default='uniform',
        metadata={"help": "noise type in coordinate noise"},
    )
    noise: float = field(
        default=1.0,
        metadata={"help": "coordinate noise for masked atoms"},
    )
    # subsample_train: float = field(
    #     default=1,
    #     metadata={"help": "shorten training set for debugging"},
    # )

@register_task("unified_pm_mlm", dataclass=MaskedLMConfig)
class ProteinMolMaskedLMTask(FairseqTask):

    cfg: MaskedLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MaskedLMConfig, dictionary_p=None, dictionary_m=None):
        super().__init__(cfg)

        dictionary_p = dictionary_p or self.load_proteins_dict(cfg)
        dictionary_m = dictionary_m or self.load_mols_dict(cfg)
        self.aa_vocab_range = (0, len(dictionary_p) - 1)
        self.mol_vocab_range = (len(dictionary_p), len(dictionary_p) + len(dictionary_m) - 1)
        self.mol_vocab_size = len(dictionary_m)
        for sym in dictionary_m:
            dictionary_p.add_symbol(sym + '_a')
        self.dictionary = dictionary_p
        self.vocab_special_list = [self.dictionary.index(c) for c in ['<s>', '<pad>', '</s>', '<unk>', "<null_1>", "<mask>"]]
        self.noise_type = cfg.noise_type
        self.pro_sample_ratio = cfg.pro_sample_ratio
        self.mol_sample_ratio = cfg.mol_sample_ratio
        self.noise = cfg.noise
        self.mask_idx = self.dictionary.index("<mask>")

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.dictionary

    @classmethod
    def setup_task(cls, cfg: MaskedLMConfig, **kwargs):
        dictionary_p = cls.load_proteins_dict(cfg)
        dictionary_m = cls.load_mols_dict(cfg)
        return cls(cfg, dictionary_p, dictionary_m)

    @classmethod
    def load_proteins_dict(cls, cfg):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict_p.txt"))
        dictionary.add_symbol("<null_1>")
        dictionary.add_symbol("<mask>")
        logger.info("Proteins dictionary: {} types".format(len(dictionary)))
        return dictionary
    
    @classmethod
    def load_mols_dict(cls, cfg):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        mol_dict = []
        with open(os.path.join(paths[0], "dict_m.txt"), 'r') as fin:
            for idx, line in enumerate(fin):
                sym = line.strip().split()[0].strip()
                mol_dict.append(sym)
        logger.info("Molecules dictionary: {} types".format(len(mol_dict)))
        return mol_dict

    def _load_protein_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split = split + '_p.lmdb'
        split_path = os.path.join(data_path, split)

        dataset = LMDBDataset(split_path) 

        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        logger.info("loaded {} proteins samples from: {}".format(len(dataset), split_path))

        return dataset
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        # return PrependTokenDataset(dataset, self.source_dictionary.bos())

    def _load_protein_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = self._load_protein_dataset_split(split, epoch, combine)

        dataset = ProteinsUnfoldDataset(
            dataset,
            'seq',
            'atoms',
            'atoms_coords',
            'atoms_name',
            unfold_prob=self.cfg.unfold_prob,
            unfold_max_len=self.cfg.tokens_per_sample,
            seed=self.cfg.seed,
            normalize_coord=True,
        )

        dataset = KeyTokenizeDataset( 
            dataset, 'unfold_seq', self.dictionary, max_seq_len=self.cfg.tokens_per_sample
        )

        dataset = RangedMaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            bos_idx=self.source_dictionary.bos(),
            eos_idx=self.source_dictionary.eos(),
            aa_vocab_range=self.aa_vocab_range,
            mol_vocab_range=self.mol_vocab_range,
            vocab_special_list=self.vocab_special_list,
            seq='unfold_seq',
            aa_mask='aa_mask', 
            coords='unfold_coords',
            noise_type=self.noise_type,
            noise=self.noise,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
            skip_masking=self.cfg.skip_masking,
        )

        dataset = ProteinsDistanceDataset(dataset, 'unfold_seq', 'aa_mask', 'unfold_coords', 'noised_coords', self.mol_vocab_size)
        
        src_dataset = KeyDataset(dataset, "unfold_seq")
        tgt_dataset = KeyDataset(dataset, "target")
        # src_coord = KeyDataset(dataset, "noised_coords")
        src_distance = KeyDataset(dataset, "noised_coords_dist")
        tgt_distance = KeyDataset(dataset, "coords_dist")
        aa_mask_dataset = KeyDataset(dataset, "aa_mask")
        src_edge_type = KeyDataset(dataset, "edge_type")

        target_dataset = RightPadDataset(
            tgt_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        input_dict = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "src_lengths": NumelDataset(src_dataset, reduce=False),
            "src_distance": RightPadDataset2D(
                src_distance,
                pad_idx=0,
            ),
            "src_edge_type": RightPadDataset2D(
                src_edge_type,
                pad_idx=0,
            ),
            "aa_mask": RightPadDataset(
                aa_mask_dataset,
                pad_idx=1,
            ),
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
                "distance_target": RightPadDataset2D(tgt_distance, pad_idx=0),
            },
            sizes=[src_dataset.sizes],
        )

        return dataset
    
    def _load_mols_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split = split + '_m.lmdb'
        split_path = os.path.join(data_path, split)

        dataset = LMDBDataset(split_path)

        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        logger.info("loaded {} molecules samples from: {}".format(len(dataset), split_path))

        return dataset

    def _load_mol_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        raw_dataset = self._load_mols_dataset_split(split, epoch, combine)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if 'train' in split:
                raw_dataset = Add2DConformerDataset(
                    raw_dataset, "smi", "atoms", "coordinates"
                )
            smi_dataset = KeyDataset(raw_dataset, "smi")
            dataset = ConformerSampleDataset(
                raw_dataset, coord_seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(raw_dataset, dataset)
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.cfg.remove_hydrogen, 
                self.cfg.remove_polar_hydrogen, 
            )
            dataset = CroppingDataset(
                dataset, self.cfg.seed, "atoms", "coordinates", self.cfg.tokens_per_sample
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.cfg.tokens_per_sample
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                mol_vocab_range=self.mol_vocab_range,
                vocab_special_list=self.vocab_special_list,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.cfg.noise_type,
                noise=self.cfg.noise,
                seed=mask_seed,
                mask_prob=self.cfg.mask_prob,
                leave_unmasked_prob=self.cfg.leave_unmasked_prob,
                random_token_prob=self.cfg.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            aa_mask_dataset = AllZerosDataset(encoder_token_dataset)

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            aa_mask_dataset = PrependAndAppend(aa_mask_dataset, 1, 1)

            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)

            print('mol src_dataset.sizes len:', len(src_dataset.sizes))

            input_dict = {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_dataset, reduce=False),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "aa_mask": RightPadDataset(
                    aa_mask_dataset,
                    pad_idx=1,
                ),
            }
            target_dataset = RightPadDataset(
                tgt_dataset, pad_idx=self.dictionary.pad()
            )
            if self.cfg.include_target_tokens:
                input_dict["target_tokens"] = target_dataset
            if self.cfg.include_index:
                input_dict["src_id"] = IdDataset()
            tgt_dict = {
                "tokens_target": target_dataset,
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
            }
            return input_dict, tgt_dict

        net_input, target = one_dataset(raw_dataset, self.cfg.seed, self.cfg.seed)
        src_dataset = net_input['src_tokens']
        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": net_input,
                "target": target['tokens_target'],
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
                "distance_target": target['distance_target'],
            },
            sizes=[src_dataset.sizes],
        )
        return dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        proteins_dataset = self._load_protein_dataset(split, epoch, combine)
        mols_dataset = self._load_mol_dataset(split, epoch, combine)

        # dataset = ConcatDataset([proteins_dataset, mols_dataset], sample_ratios=[self.pro_sample_ratio, self.mol_sample_ratio])
        dataset = mols_dataset

        print('mols_dataset len:', len(mols_dataset))
        print('dataset.sizes len:', len(dataset.sizes))
        
        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(dataset))
        print('shuffle len:', len(shuffle))
        self.datasets[split] = SortDataset(
            dataset, sort_order=[shuffle, dataset.sizes]
        )

        logger.info("totally loaded {} samples for {} set".format(len(dataset), split))

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
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
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
