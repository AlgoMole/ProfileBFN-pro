export CUDA_VISIBLE_DEVICES=0

task=$1
config_name=$2

save_name_suffix=$3

config=represent_learning/config/${task}/${config_name}.yaml

declare -u upper_config_name=$config_name


save_path=./checkpoints/${task}_train${task}_${upper_config_name}

tensorboard_dir=./tensorboard/${task}_${upper_config_name}

python represent_learning/testing.py --config ${config} \
    --model.save_path ${save_path} \
    --model_checkpoint.dirpath ${save_path} \
    --Trainer.logger tensorboard 

