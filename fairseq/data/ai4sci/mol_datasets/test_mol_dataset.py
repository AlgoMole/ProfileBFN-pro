from lmdb_dataset import LMDBDataset
from fairseq.data import Dictionary
from add_2d_conformer_dataset import Add2DConformerDataset
from all_zeros_dataset import AllZerosDataset
from lmdb_dataset import LMDBDataset
from key_dataset import KeyDataset
from conformer_sample_dataset import ConformerSampleDataset
from atom_type_dataset import AtomTypeDataset
from remove_hydrogen_dataset import RemoveHydrogenDataset
from cropping_dataset import CroppingDataset
from normalize_dataset import NormalizeDataset
from tokenize_dataset import TokenizeDataset
from mask_points_dataset import MaskPointsDataset
from prepend_token_dataset import PrependTokenDataset
from append_token_dataset import AppendTokenDataset
from distance_dataset import EdgeTypeDataset, DistanceDataset
from from_numpy_dataset import FromNumpyDataset
from coord_pad_dataset import RightPadDatasetCoord, RightPadDatasetCross2D, RightPadDataset2D
from fairseq.data import RightPadDataset

data_path = '/data1/zkj/datasets/ligands/small_test.lmdb/data.mdb'

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
seed = 1
tokens_per_sample = 1024

for sym in mol_dict:
    dictionary.add_symbol(sym + '_a')

mask_idx = dictionary.index('<mask>')

raw_dataset = LMDBDataset(data_path)
max_seq_len = 1024

raw_dataset = Add2DConformerDataset(
    raw_dataset, "smi", "atoms", "coordinates"
)
smi_dataset = KeyDataset(raw_dataset, "smi")
dataset = ConformerSampleDataset(
    raw_dataset, seed, "atoms", "coordinates"
)
dataset = AtomTypeDataset(raw_dataset, dataset)
dataset = RemoveHydrogenDataset(
    dataset,
    "atoms",
    "coordinates",
    True,
    True,
)
dataset = CroppingDataset(
    dataset, seed, "atoms", "coordinates", tokens_per_sample
)
dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
token_dataset = KeyDataset(dataset, "atoms")
token_dataset = TokenizeDataset(
    token_dataset, dictionary, max_seq_len=tokens_per_sample
)
coord_dataset = KeyDataset(dataset, "coordinates")
expand_dataset = MaskPointsDataset(
    token_dataset,
    coord_dataset,
    dictionary,
    mol_vocab_range=mol_vocab_range,
    vocab_special_list=vocab_special_list,
    pad_idx=dictionary.pad(),
    mask_idx=mask_idx,
    noise_type='uniform',
    noise=1.0,
    seed=seed,
    mask_prob=0.15,
    leave_unmasked_prob=0.1,
    random_token_prob=0.1,
)

def PrependAndAppend(dataset, pre_token, app_token):
    dataset = PrependTokenDataset(dataset, pre_token)
    return AppendTokenDataset(dataset, app_token)

encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
encoder_target_dataset = KeyDataset(expand_dataset, "targets")
encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

aa_mask_dataset = AllZerosDataset(encoder_token_dataset)

src_dataset = PrependAndAppend(
    encoder_token_dataset, dictionary.bos(), dictionary.eos()
)
tgt_dataset = PrependAndAppend(
    encoder_target_dataset, dictionary.pad(), dictionary.pad()
)
aa_mask_dataset = PrependAndAppend(aa_mask_dataset, 1, 1)

encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
coord_dataset = FromNumpyDataset(coord_dataset)
coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
distance_dataset = DistanceDataset(coord_dataset)

src_dataset = RightPadDataset(
    src_dataset,
    pad_idx=dictionary.pad(),
)

src_distance = RightPadDataset2D(
    encoder_distance_dataset,
    pad_idx=0,
)

print('src_dataset.sizes len:', len(src_dataset.sizes))
exit()
encoder_token_dataset.set_epoch(0)
# for ind, data in enumerate(encoder_coord_dataset):
# for ind, data in enumerate(encoder_token_dataset):
data_to_collect = []
for ind, data in enumerate(src_distance):
    data_to_collect.append(data)
    if ind == 5:
        break
print(src_distance.collater(data_to_collect))
print(src_distance.collater(data_to_collect).size())