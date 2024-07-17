#!/bin/bash

# Change directory to the root of mmsegmentation
cd mmsegmentation

# Define the paths to your PyTorch scripts
dataset_name=siim
base_work_dir="./experiments/${dataset_name}/"

# List of seeds
seeds=(42 0 1)

# Array of models and their corresponding configurations
declare -a MODELS=(
    "ConVIRT upernet_r50"
    "GLoRIA upernet_r50"
    "MedCLIP_R50 upernet_r50"
    "MedCLIP_ViT upernet_swin_medclip"
    "MedKLIP upernet_medklip"
    "PTUnifier upernet_ptunifier"
    "MFLAG upernet_r50"
    "MGCA_R50 upernet_r50"
    "MGCA_ViT upernet_vit"
    "MRM upernet_vit"
    "REFERS upernet_vit"
)

# Loop over models and print Python command
for model in "${MODELS[@]}"; do
    set -- $model
    model_name=$1
    config_name=$2
    shift 2

    config="configs/benchmark/$dataset_name/$config_name.py"

    # Run the script with different data splits and seeds
    for seed in "${seeds[@]}"; do
        work_dir="${base_work_dir}${model_name}_${seed}/"

        command="python tools/train.py $config --work-dir $work_dir --cfg-options randomness.seed=$seed"

        # Print the command
        echo "Running command: $command"
        
        # Uncomment the line below to execute the command
        eval "$command"
    done
done
