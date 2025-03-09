SHELL := /bin/bash

current_dir :=$(shell pwd)

CKPT_PATH:= /sharefs/yupei/ProfileBFN-pro/checkpoints/ProfileBFN/ProfileBFN_650M/checkpoint_best.pt

sample_sequence:
	${eval DATADIR:=./data/CAMEO}
	$(eval OUT_DIR:=./results/$@)
	rm -rf ${OUT_DIR}
	cp -r ${DATADIR} ${OUT_DIR}
	for p in `ls ${OUT_DIR}/*.fasta`; do \
		pp=$${p:0:-6}; \
		python seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 10 --batch-size 5 \
				--input-fasta $${pp}.fasta --output-a3m $${pp}.a3m.gen; \
	done

sample_profile:
	${eval DATADIR:=./data/CAMEO}
	$(eval OUT_DIR:=./results/$@)
	rm -rf ${OUT_DIR}
	cp -r ${DATADIR} ${OUT_DIR}
	for p in `ls ${OUT_DIR}/*.a3m`; do \
		pp=$${p:0:-4}; \
		python profile_seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 1000 --batch-size 50 \
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


