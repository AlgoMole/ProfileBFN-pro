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


