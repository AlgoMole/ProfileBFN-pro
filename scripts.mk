SHELL := /bin/bash

current_dir :=$(shell pwd)

CKPT_PATH:= /sharefs/gongjj/accounting/gongjj_dev_bfn4prot/bfn_650M_b2m_fromESM_16a100/27ea3bd_18159/checkpoint_31_320000.pt

sample_profile:
	${eval DATADIR:=./data/CAMEO}
	$(eval OUT_DIR:=./results/$@)
	rm -rf ${OUT_DIR}
	cp -r ${DATADIR} ${OUT_DIR}
	for p in `ls ${OUT_DIR}/*.a3m`; do \
		pp=$${p:0:-4}; \
		python profile_seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4 --batch-size 2 \
				--input-a3m $${pp}.a3m --output-a3m $${pp}.a3m; \
	done

sample_enzyme:
	${eval DATADIR:=./data/Enzyme}
	$(eval OUT_DIR:=./results/$@)
	rm -rf ${OUT_DIR}
	cp -r ${DATADIR} ${OUT_DIR}
	for p in `ls ${OUT_DIR}/*.a3m`; do \
		pp=$${p:0:-4}; \
		python profile_seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4 --batch-size 2 \
				--input-a3m $${pp}.a3m --output-a3m $${pp}.a3m; \
	done

sample_lysozyme:
	${eval DATADIR:=./data/Lysozyme}
	$(eval OUT_DIR:=./results/$@)
	rm -rf ${OUT_DIR}
	cp -r ${DATADIR} ${OUT_DIR}
	for p in `ls ${OUT_DIR}/*.a3m`; do \
		pp=$${p:0:-4}; \
		python profile_seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4 --batch-size 2 \
				--input-a3m $${pp}.a3m --output-a3m $${pp}.a3m; \
	done

# Represent-Learning
finetune_represent_learning:
# TASK choosing from Thermostability, HumanPPI, MetalIonBinding, EC, GO/MF, GO/CC, GO/, DeepLoc/cls2, DeepLoc/cls10
	${eval TASK:= Thermostability}  
# MODEL choosing from ProfileBFN_150M, ProfileBFN_650M
	${eval MODEL:= ProfileBFN_650M}
	bash ./represent_learning/run_train.sh ${TASK} ${MODEL} train${TASK}_${MODEL}


# hyper-params
PROJECT_NAME := gongjj_dev_bfn4prot
DATAPATH := '/sharefs/gongjj/data/uniref50_202403/epoch_*'
SAVEROOT := ${current_dir}/results
GIT_KEY := /sharefs/gongjj/.ssh/id_rsa

ENC_LAYER := 33
ENC_HEAD := 20
ENC_DIM := 1280
FFN_DIM := 5120
MAX_TOKENS := 4096
UPDATE_FREQ := 64

# try trainning
ProfileBFN_650M_train:
	$(eval EXP_NAME:= $@) 
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	${eval COMMIT := ${shell git rev-parse HEAD}}
	_max_update=500000; \
	loss_type=bfl2; \
	clip_norm=1.0; \
	total_num_update=$$(($${_max_update}*9/10)); \
	max_update=270000; \
	for bf_type in mbcltbf; do \
		for diff_accuracy in mtps0.9; do \
			for beta1 in 1.6; do \
				code_=${REVISION}_$${RANDOM}; \
				export MKL_SERVICE_FORCE_INTEL=1; \
				export MKL_THREADING_LAYER=GNU; \
				export PYTHONIOENCODING=utf-8; \
				export WANDB_NAME=${EXP_NAME}_$${code_}; \
				mkdir ${SAVEROOT}/${EXP_NAME}/$${code_}; \
				python train.py ${DATAPATH} \
					--revision ${COMMIT} \
					--save-dir ${SAVEROOT}/${EXP_NAME}/$${code_} \
					--finetune-from-model '/sharefs/gongjj/data/torch_home/esm2_t33_650M_UR50D.pt' \
					--train-subset train50 \
					--valid-subset valid50 \
					--skip-invalid-size-inputs-valid-test \
					--ignore-unused-valid-subsets \
					--shorten-method "random_crop" \
					--shorten-data-split-list "train50,valid50" \
					--tokens-per-sample 1024 \
					--max-positions 1024 \
					--encoder-normalize-before \
					--dataset-impl fasta \
					--ddp-backend=c10d --fp16 \
					--task p_bfn_lm \
					--criterion bfn_lm \
					--arch bfn_roberta \
					--num-workers 10 \
					--mask-prob 0.15 \
					--encoder-layers ${ENC_LAYER} \
					--encoder-attention-heads ${ENC_HEAD} \
					--encoder-embed-dim ${ENC_DIM} \
					--encoder-ffn-embed-dim ${FFN_DIM} \
					--attention-dropout 0.0 \
					--activation-dropout 0.0 \
					--dropout 0.0 \
					--sample-break-mode "eos" \
					--optimizer adam --adam-betas "(0.9,0.98)" \
					--lr 4e-4 --lr-scheduler polynomial_decay \
					--end-learning-rate 4e-05 \
					--warmup-updates 2000 \
					--max-update $${max_update} \
					--total-num-update $${total_num_update} \
					--weight-decay 0.01 --clip-norm $${clip_norm} \
					--max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
					--max-tokens-valid ${MAX_TOKENS} \
					--log-interval 1 \
					--fixed-validation-seed 7 \
					--seed 1 \
					--save-interval-updates 2000 \
					--no-epoch-checkpoints \
					--fp16-scale-tolerance 0.0 \
					--fp16-init-scale 4 \
					--min-loss-scale 0.0001 \
					--n-valid-samples 5000 \
					--data-buffer-size 4 \
					--beta1 $${beta1} \
					--time-dim 1 \
					--beta-time-order 1.0 \
					--loss-type $${loss_type} \
					--bf-type $${bf_type} \
					--diff-accuracy $${diff_accuracy}; \
			done; \
		done; \
	done; \