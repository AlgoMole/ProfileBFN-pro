workdir=$(cd $(dirname $0); pwd)

CUDA_VISIBLE_DEVICES=4,5,6,7    ccmpred/scripts/run_evaluate.sh -i $workdir/results/sample_profile -o $workdir/results/sample_profile_ccmpred