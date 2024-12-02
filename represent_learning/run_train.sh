export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

task=$1
config_name=$2

save_name_suffix=$3
save_name=${task}_${save_name_suffix}

config=represent_learning/config/${task}/${config_name}.yaml
save_path=checkpoints/${save_name}


mkdir -p ./wandb

python represent_learning/training.py --config ${config} \
    --model.save_path ${save_path} \
    --model_checkpoint.dirpath ${save_path} \
    --Trainer.logger wandb

