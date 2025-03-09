# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from fairseq.data import BaseWrapperDataset
import numpy as np
import torch
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import List  
import logging
logger = logging.getLogger(__name__)

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# empty_data_dict = {
#     'seq': '',
#     'atoms': [],
#     'atoms_coords': [],
#     'atoms_name': [],
# }

@njit
def get_unfold_seq(seq, seq_len, unfold_idc, atoms, atoms_coords, normalize_coord):
    unfold_len = seq_len
    offset_list = [0 for _ in range(seq_len)]
    for idc in unfold_idc:
        unfold_len += len(atoms[idc])
        offset_list[idc] += len(atoms[idc])
    unfold_coords = [[0.0, 0.0, 0.0] for _ in range(unfold_len)]
    aa_mask = [1] * unfold_len
    if len(unfold_idc) == 0:
        return [t for t in seq], aa_mask, unfold_coords
    unfold_seq = ['__' for _ in range(unfold_len)] 
    offset = 0
    for i in range(seq_len):
        unfold_seq[i + offset] = seq[i]
        offset += offset_list[i]
    aa_id = 0
    atom_id = 0
    mean_atoms_coords = [0.0, 0.0, 0.0]
    for i in range(unfold_len):
        if unfold_seq[i] == '__':
            if normalize_coord and atom_id == 0:
                for j in range(len(atoms[aa_id - 1])):
                    for co in range(3):
                        mean_atoms_coords[co] += atoms_coords[aa_id - 1][j][co]
                for co in range(3):
                    mean_atoms_coords[co] /= len(atoms[aa_id - 1])
            unfold_seq[i] = atoms[aa_id - 1][atom_id] + '_a'
            for co in range(3):
                unfold_coords[i][co] = atoms_coords[aa_id - 1][atom_id][co] - mean_atoms_coords[co]
            aa_mask[i] = 0
            atom_id += 1
        else:
            aa_id += 1
            atom_id = 0
            mean_atoms_coords = [0.0, 0.0, 0.0]
    return unfold_seq, aa_mask, unfold_coords

class ProteinsUnfoldDataset(BaseWrapperDataset): # update sizes
    def __init__(
        self,
        dataset,
        seq,
        atoms,
        atoms_coords,
        atoms_name,
        unfold_prob=0.0,
        unfold_max_len=0,
        seed=1,
        normalize_coord=True,
    ):
        self.dataset = dataset
        self.unfold_prob = unfold_prob
        self.seq = seq
        self.atoms = atoms
        self.atoms_coords = atoms_coords
        self.atoms_name = atoms_name
        self.seed = seed
        self.epoch = 0
        self.unfold_max_len = unfold_max_len
        self.set_epoch(0)
        self.unfold_idc_list = None
        self.normalize_coord = normalize_coord

    def update_sizes(self):
        self._sizes = self.dataset.sizes
        return
        logger.info("Updating dataset sizes.")
        new_sizes = []
        new_unfold_idc = []
        for index in range(len(self.dataset)):
            seed = int(hash((self.seed, self.epoch, index)) % 1e6)
            rng = np.random.default_rng(seed)
            protein_seq = self.dataset[index][self.seq]
            atoms = self.dataset[index][self.atoms]
            aa_lens = np.array([len(aa) for aa in atoms])
            sz = len(protein_seq)
            assert sz > 0
            num_unfold = int(
                self.unfold_prob * sz
                + rng.random()
            )
            unfold_idc = rng.choice(sz, num_unfold, replace=False)
            while len(unfold_idc) > 0 and np.sum(aa_lens[unfold_idc]) + sz > self.unfold_max_len:
                unfold_idc = unfold_idc[:-1]
            new_sizes.append(np.sum(aa_lens[unfold_idc]) + sz)
            new_unfold_idc.append(np.array(unfold_idc))
        self._sizes = np.array(new_sizes)
        self.unfold_idc_list = new_unfold_idc
        
    @property
    def sizes(self):
        return self._sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
        if self.epoch is not None:
            self.update_sizes()

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False  # the item sizes changed

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        protein_seq = self.dataset[index][self.seq]
        atoms = self.dataset[index][self.atoms]
        atoms_coords = self.dataset[index][self.atoms_coords]
        sz = len(protein_seq)
        if self.unfold_idc_list is not None:
            unfold_idc = self.unfold_idc_list[index]
        else:
            seed = int(hash((seed, epoch, index)) % 1e6)
            rng = np.random.default_rng(seed)
            aa_lens = np.array([len(aa) for aa in atoms])
            num_unfold = int(
                self.unfold_prob * sz
                + rng.random()
            )
            unfold_idc = rng.choice(sz, num_unfold, replace=False)
            while len(unfold_idc) > 0 and np.sum(aa_lens[unfold_idc]) + sz > self.unfold_max_len:
                unfold_idc = unfold_idc[:-1]
            unfold_idc = np.array(unfold_idc)
        protein_seq = List(protein_seq)
        atoms = List(atoms)
        atoms_coords = List(atoms_coords)
        unfold_seq, aa_mask, unfold_coords = get_unfold_seq(protein_seq, sz, unfold_idc, atoms, atoms_coords, self.normalize_coord)
        ret = {
            'unfold_seq': unfold_seq,
            'aa_mask': np.array(aa_mask, dtype=bool),
            'unfold_coords': np.array(unfold_coords),
        }
        return ret
