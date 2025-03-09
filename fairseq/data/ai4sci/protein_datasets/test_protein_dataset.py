from lmdb_dataset import LMDBDataset
from fairseq.data import Dictionary
from proteins_unfold_dataset import ProteinsUnfoldDataset
from key_tokenize_dataset import KeyTokenizeDataset
from ranged_mask_tokens_dataset import RangedMaskTokensDataset
from protein_distance_dataset import ProteinsDistanceDataset
from key_dataset import KeyDataset
from coord_pad_dataset import RightPadDatasetCoord, RightPadDatasetCross2D, RightPadDataset2D
from fairseq.data import RightPadDataset

data_path = '/data1/zkj/AFDB_HC_small/small.lmdb'

dictionary = Dictionary.load('/mnt/data0/zkj/datasets/uniref50/dict.txt')
dictionary.add_symbol("<null_1>")
dictionary.add_symbol("<mask>")

mol_dict = []
with open("/data1/zkj/AFDB_HC_small/mole_dict.txt", 'r') as fin:
    for idx, line in enumerate(fin):
        sym = line.strip().split()[0].strip()
        mol_dict.append(sym)

aa_vocab_range = (0, len(dictionary) - 1)
mol_vocab_range = (len(dictionary), len(dictionary) + len(mol_dict) - 1)
mol_vocab_size = len(mol_dict)
vocab_special_list = [dictionary.index(c) for c in ['<s>', '<pad>', '</s>', '<unk>', "<null_1>", "<mask>"]]

for sym in mol_dict:
    dictionary.add_symbol(sym + '_a')

dataset = LMDBDataset(data_path)
max_seq_len = 1024

dataset = ProteinsUnfoldDataset(
    dataset,
    'seq',
    'atoms',
    'atoms_coords',
    'atoms_name',
    unfold_prob=0.1,
    seed=1,
    unfold_max_len=max_seq_len,
)

dataset = KeyTokenizeDataset( # TODO
    dataset, 'unfold_seq', dictionary, max_seq_len=max_seq_len
)

dataset = RangedMaskTokensDataset.apply_mask(
    dataset,
    dictionary,
    pad_idx=dictionary.pad(),
    mask_idx=dictionary.index('mask'),
    bos_idx=dictionary.bos(),
    eos_idx=dictionary.eos(),
    aa_vocab_range=aa_vocab_range,
    mol_vocab_range=mol_vocab_range,
    vocab_special_list=vocab_special_list,
    seq='unfold_seq',
    aa_mask='aa_mask', 
    coords='unfold_coords',
    noise_type='uniform',
    noise=1.0,
    seed=1,
    mask_prob=0.2,
    leave_unmasked_prob=0.1,
    random_token_prob=0.1,
    mask_multiple_length=1,
    mask_stdev=0.0,
    skip_masking=False,
)

dataset = ProteinsDistanceDataset(dataset, 'unfold_seq', 'aa_mask', 'unfold_coords', 'noised_coords', mol_vocab_size)

src_dataset = KeyDataset(dataset, "unfold_seq")
tgt_dataset = KeyDataset(dataset, "target")
src_distance = KeyDataset(dataset, "noised_coords_dist")
tgt_distance = KeyDataset(dataset, "coords_dist")
aa_mask_dataset = KeyDataset(dataset, "aa_mask")
src_edge_type = KeyDataset(dataset, "edge_type")

src_dataset = RightPadDataset(
    src_dataset,
    pad_idx=dictionary.pad(),
)

src_distance = RightPadDataset2D(
    src_distance,
    pad_idx=0,
)

data_to_collect = []
for ind, data in enumerate(src_distance):
    # print(data)
    data_to_collect.append(data)
    if ind == 5:
        break
print(src_distance.collater(data_to_collect))
print(src_distance.collater(data_to_collect).size())
# for ind, data in enumerate(dataset):
#     print(data.keys())
#     print(data['unfold_seq'])
#     print(data['aa_mask'])
#     print(data['target'])
#     print(data['edge_type'])
#     # print(data['unfold_coords'])
#     if ind == 2:
#         exit()