# ProfileBFN
Official implementation of ICLR 2025 ["ProfileBFN: Steering Protein Family Design through Profile Bayesian Flow"](https://openreview.net/forum?id=PSiijdQjNU&noteId=sRV2quHqPd).

## Environment
The environment is based on PyTorch 1.13. Follow the [official installation instructions](https://pytorch.org/get-started/previous-versions/) to set it up according to your CUDA version. Then, install the following packages:

```bash
pip install omegaconf hydra-core bitarray rdkit-pypi scipy lmdb numba scikit-learn
```

More detailed environment settings are located in env.yaml


-----

## Data
Data used for evaluating the model is already put in the `data` folder

---


## Checkpoints
We provide the pretrained checkpoint as [ProfileBFN_150M.ckpt](https://huggingface.co/hhhhhhh789/ProfileBFN_150M) and [ProfileBFN_650M.ckpt](https://huggingface.co/hhhhhhh789/ProfileBFN_650M), please download all files and set the CKPT_PATH to the corresponding directory. 


## Sampling
`mkdir ./results` All Generation Results will be placed in such subdir.

Run `make sample_profile -f scripts.mk` to sample protein family based MSA. Note that inputs with inconsistent lengths would be automatically aligned.

Run `make sample_sequence -f scripts.mk` to sample protein family based on single protein sequence.


## Evaluation
### Evaluating generated protein family by CCMPRED
Clone [CCMPRED](https://github.com/jingjing-gong/contact_evaluation) repo in dir `test/ccmpred` and follow instructions as their README.  

targets are generated sequence under `results/sample_profile` dir after the sampling process
```bash
cd test/ccmpred
docker build -f docker/Dockerfile -t exp/contact_evaluation .
CUDA_VISIBLE_DEVICES=4,5,6,7 ./scripts/run_evaluate.sh -i <input_dir> -o <output_dir>
```


## Citation
```bash
@article{gong2025steering,
  title={Steering Protein Family Design through Profile Bayesian Flow},
  author={Gong, Jingjing and Pei, Yu and Long, Siyu and Song, Yuxuan and Zhang, Zhe and Huang, Wenhao and Cao, Ziyao and Zhang, Shuyi and Zhou, Hao and Ma, Wei-Ying},
  journal={arXiv preprint arXiv:2502.07671},
  year={2025}
}

