# ProfileBFN
Official implementation of ICLR 2025 ["ProfileBFN: Steering Protein Family Design through Profile Bayesian Flow"](https://openreview.net/forum?id=PSiijdQjNU&noteId=sRV2quHqPd).

## Environment
It is highly recommended to install via docker if a Linux server with NVIDIA GPU is available.

Otherwise, you might check [README for env](docker/README.md) for further details of docker or conda setup.

### Prerequisite

### Install via Docker
We highly recommend you to set up the environment via docker, since all you need to do is a simple `make` command.
```bash
cd ./docker
make
```


-----

## Data
Data used for evaluating the model is already put in the `data` folder

To train the model with Uniref data, please download them in the `uniref` folder

For data needed of CLEAN model, please download `split100.csv`, `split100.fasta` under `data/CLEAN` directory

For data used as represent learning, please refer to the same [dataset](https://drive.google.com/drive/folders/11dNGqPYfLE3M-Mbh4U7IQpuHxJpuRr4g?usp=sharing) as SaProt and DPLM and put it into `data/LMDB` folder

---


## Checkpoints
We provide the pretrained checkpoint as [ProfileBFN_650M.ckpt](). 

### (Optional) Download download [CLEAN]() model for enzyme functional classification


### (Optional) Download ESMFold for Lysozyme evaluation

---


## Training
Run `make -f scripts.mk ProfileBFN_650M_train` (without the need for data preparation)



## Sampling

Run `make evaluate -f scripts.mk` to sample family sequences


## Evaluation
### Evaluating generated protein family by CCMPRED
Clone [CCMPRED](https://github.com/jingjing-gong/contact_evaluation) repo in dir `test/ccmpred` and follow instructions as README.md for ccmpred.  

targets are generated sequence under `results/sample_profile` dir after the sampling process
```bash
cd test/ccmpred
docker build -f docker/Dockerfile -t exp/contact_evaluation .
CUDA_VISIBLE_DEVICES=4,5,6,7 ./scripts/run_evaluate.sh -i <input_dir> -o <output_dir>
```


### Evaluating generated enzyme family by CLEAN model
For Accuracy, please follow instructions of CLEAN



### Evaluating generated Lysozyme family by ESMFold


## Citation


