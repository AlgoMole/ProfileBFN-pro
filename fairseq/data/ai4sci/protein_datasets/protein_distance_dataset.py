import numpy as np
import torch
from scipy.spatial import distance_matrix
from functools import lru_cache
from fairseq.data import BaseWrapperDataset


class ProteinsDistanceDataset(BaseWrapperDataset):
    def __init__(self, dataset, seq, aa_mask, coords, noised_coords, num_types):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_types = num_types
        self.aa_mask = aa_mask
        self.seq = seq
        self.coords = coords
        self.noised_coords = noised_coords

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        new_item = item.copy()
        coords = item[self.coords].view(-1, 3)
        aa_mask = item[self.aa_mask].long()
        noised_coords = item[self.noised_coords]
        # aa_mask = (torch.sum(coords ** 2, dim=1) == 0).long()

        mol_mask = 1 - aa_mask
        same_aa_mask = torch.cumsum(aa_mask, dim=0)
        same_aa_mask[aa_mask == 1] = -1
        not_mol_mat = ~ (torch.outer(mol_mask, mol_mask).bool().numpy())
        not_same_aa_mat = (same_aa_mask.view(-1, 1) != same_aa_mask.view(1, -1)).bool().numpy()
        not_valid_mol_mask = not_mol_mat + not_same_aa_mat

        coords = coords.numpy()
        coords_dist = distance_matrix(coords, coords).astype(np.float32)
        coords_dist[not_valid_mol_mask] = -1

        noised_coords = noised_coords.numpy()
        noised_coords_dist = distance_matrix(noised_coords, noised_coords).astype(np.float32)
        noised_coords_dist[not_valid_mol_mask] = -1

        seq = item[self.seq]
        edge_type = (seq.view(-1, 1) * self.num_types + seq.view(1, -1)).long()
        not_valid_mol_mask = torch.from_numpy(not_valid_mol_mask)
        edge_type.masked_fill_(not_valid_mol_mask, -1)      

        new_item['coords_dist'] = torch.from_numpy(coords_dist)
        new_item['noised_coords_dist'] = torch.from_numpy(noised_coords_dist)
        new_item['edge_type'] = edge_type

        return new_item