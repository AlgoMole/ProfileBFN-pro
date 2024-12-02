SHELL := /bin/bash

CKPT_PATH:="/sharefs/gongjj/accounting/gongjj_dev_bfn4prot/bfn_150M_b2m_8a100/e73a7cd_10016/checkpoint_best.pt"
MLM_CKPT_PATH:="/sharefs/gongjj/accounting/gongjj_dev_bfn4prot/mlm_150M_b128k_8a100/f60c726_24642/checkpoint_best.pt"
DATADIR:="/sharefs/gongjj/data/bfn4prot_eval"
UNCOND_DATADIR:="/sharefs/yupei/BFN4Prot/sample_data/bfn4prot_unconditional"
UNCOND_MLM_DATADIR:="/sharefs/yupei/BFN4Prot/sample_data/mlm_uncond"
MOTIF_DATADIR:="/sharefs/yupei/BFN4Prot/sample_data/bfn4prot_motif"
esm_model_dir="/sharefs/yupei/project/playground/dplm/airkingbd_download/EsmFold"
MQ:="bfn4prot_eval_cond"
UMQ:="bfn4prot_eval_uncond"
NAME:="test_bfn_std_uncond"
SEED:="L_seeds"

max_tokens=1024

generate:
	for p in `ls ${DATADIR}/*.pdb`; do \
		pp=$${p:0:-4};\
		echo "python eval/exp/seqgen.py --ckpt-path ${CKPT_PATH} \
			--input-fasta $${pp}.fasta --output-a3m $${pp}.a3m" | python experiments/sendtask2queue.py -q ${MQ} -p 0; \
	done
debug:
	$(eval DATADIR:=/sharefs/gongjj/data/debug_bfn4prot_eval)
	for p in `ls ${DATADIR}/*.pdb`; do \
		pp=$${p:0:-4};\
		python eval/exp/seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4096 --batch-size 32\
			--input-fasta $${pp}.fasta --output-a3m $${pp}.a3m; \
	done

debug2:
	$(eval DATADIR:=/sharefs/gongjj/data/debug2_bfn4prot_eval)
	for p in `ls ${DATADIR}/*.pdb`; do \
		pp=$${p:0:-4};\
		python eval/exp/seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4096 --batch-size 32\
			--input-fasta $${pp}.fasta --output-a3m $${pp}.a3m; \
	done

# running this
debug3:
	$(eval DATADIR:=/sharefs/gongjj/data/debug3_bfn4prot_eval)
	for p in `ls ${DATADIR}/*.pdb`; do \
		pp=$${p:0:-4};\
		python eval/exp/seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4096 --batch-size 32\
			--input-fasta $${pp}.fasta --output-a3m $${pp}.a3m; \
	done

debug4:
	$(eval DATADIR:=/sharefs/gongjj/data/debug4_bfn4prot_eval)
	for p in `ls ${DATADIR}/*.pdb`; do \
		pp=$${p:0:-4};\
		python eval/exp/seqgen.py --ckpt-path ${CKPT_PATH} --num-seqs 4096 --batch-size 32\
			--input-fasta $${pp}.fasta --output-a3m $${pp}.a3m; \
	done


generate_uncond:
	rm -rf ${UNCOND_DATADIR}/${NAME}
	mkdir -p ${UNCOND_DATADIR}/${NAME}
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	for l in 100 200 300 400 500 600 700 800 900 1000; do \
		python eval/exp/seqgen_unconditional.py --ckpt-path ${CKPT_PATH} \
			--input-fasta ${UNCOND_DATADIR}/len_$${l}.fasta --output-a3m ${UNCOND_DATADIR}/${NAME}/iter_500_L_$${l}.fasta --num-seqs 40 --batch-size 10; \
	done
	bash eval/dplm/analysis/plddt_calculate.sh ${UNCOND_DATADIR}/${NAME} 
	python eval/dplm/analysis/uncond_analysis.py --output_dir ${UNCOND_DATADIR}/${NAME}

debug_uncond:
	rm -rf ${UNCOND_DATADIR}/${NAME}_debug
	mkdir -p ${UNCOND_DATADIR}/${NAME}_debug
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	for l in 100 200 300 400 500; do \
		python eval/exp/seqgen_unconditional.py --ckpt-path ${CKPT_PATH} \
			--input-fasta ${UNCOND_DATADIR}/${SEED}/len_$${l}.fasta --output-a3m ${UNCOND_DATADIR}/${NAME}_debug/iter_500_L_$${l}.fasta --model_arch bfn --prior rand --num-seqs 10 --max_iters 500 --start_t 0.0 --batch-size 5; \
	done
	bash eval/dplm/analysis/plddt_calculate.sh ${UNCOND_DATADIR}/${NAME}_debug
	python eval/dplm/analysis/uncond_analysis.py --output_dir ${UNCOND_DATADIR}/${NAME}_debug

debug_uncond_mlm:
	rm -rf ${UNCOND_MLM_DATADIR}/${NAME}_debug
	mkdir -p ${UNCOND_MLM_DATADIR}/${NAME}_debug
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	for l in 100 200 300 400 500; do \
		python eval/exp/seqgen_mlm_uncond.py --ckpt-path ${MLM_CKPT_PATH} \
			--input-fasta ${UNCOND_MLM_DATADIR}/${SEED}/len_$${l}.fasta --output-a3m ${UNCOND_MLM_DATADIR}/${NAME}_debug/iter_500_L_$${l}.fasta --prior rand --num-seqs 10 --max_iters 500 --start_t 0.0 --batch-size 5; \
	done
	bash eval/dplm/analysis/plddt_calculate.sh ${UNCOND_MLM_DATADIR}/${NAME}_debug
	python eval/dplm/analysis/uncond_analysis.py --output_dir ${UNCOND_MLM_DATADIR}/${NAME}_debug

generate_motif:
	rm -rf ${MOTIF_DATADIR}/${NAME}
	mkdir -p ${MOTIF_DATADIR}/${NAME}/scaffold_fasta
	mkdir -p ${MOTIF_DATADIR}/${NAME}/scaffold_info
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	python eval/exp/seqgen_motif.py --ckpt-path ${CKPT_PATH} --prior rand --batch-size 1 \
			--input_dir ${MOTIF_DATADIR}/scaffolding-pdbs/ --output_dir ${MOTIF_DATADIR}/${NAME}/ \
			--max_iters 500 --num-seqs 100 \
			--sample_strategy linear \
			--start_t 0.6 --temperature 0.0 ; \
	
	python eval/dplm/analysis/cal_plddt_dir.py -i ${MOTIF_DATADIR}/${NAME}/scaffold_fasta -o ${MOTIF_DATADIR}/${NAME}/scaffold_fasta/esmfold_pdb --max-tokens-per-batch ${max_tokens} -m ${esm_model_dir}
	python eval/dplm/analysis/motif_analysis.py --scaffold_dir ${MOTIF_DATADIR}/${NAME}

debug_motif:
	rm -rf ${MOTIF_DATADIR}/${NAME}
	mkdir -p ${MOTIF_DATADIR}/${NAME}/scaffold_fasta
	mkdir -p ${MOTIF_DATADIR}/${NAME}/scaffold_info
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	python eval/exp/seqgen_motif.py --ckpt-path ${CKPT_PATH} --prior rand --batch-size 1 \
			--input_dir ${MOTIF_DATADIR}/scaffolding-pdbs/ --output_dir ${MOTIF_DATADIR}/${NAME}/ \
			--max_iters 500 --num-seqs 10 \
			--single_pdb 1prw --sample_strategy linear \
			--start_t 0.6 --temperature 0.0 ; \
	
	python eval/dplm/analysis/cal_plddt_dir.py -i ${MOTIF_DATADIR}/${NAME}/scaffold_fasta -o ${MOTIF_DATADIR}/${NAME}/scaffold_fasta/esmfold_pdb --max-tokens-per-batch ${max_tokens} -m ${esm_model_dir}
	python eval/dplm/analysis/motif_analysis.py --scaffold_dir ${MOTIF_DATADIR}/${NAME}  --single_pdb 1prw
