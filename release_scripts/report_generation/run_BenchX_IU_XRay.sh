#!/bin/bash

dataset_name=IUXray

# List of seeds
seeds=(42 0 1)

# Array of models and their corresponding configurations
declare -a MODELS=(
    "ConVIRT convirt"
    "GLoRIA gloria"
    "MedCLIP_R50 medclip_rn50"
    "MedCLIP_ViT medclip_vit"
    "MedKLIP medklip"
    "PTUnifier ptunifier"
    "MFLAG mflag"
    "MGCA_R50 mgca_rn50"
    "MGCA_ViT mgca_vit"
    "MRM mrm"
    "REFERS refers"
)

# Loop over models and print Python command
for model in "${MODELS[@]}"; do
    set -- $model
    model_name=$1
    config_name=$2
    shift 2

    config="configs/report_generation/$dataset_name/$config_name.yml"

    # Run the script with different data splits and seeds
    for seed in "${seeds[@]}"; do
        command="python bin/train.py $config name=$model_name seed=${seed}"

        # Print the command
        echo "Running command: $command"
        
        # Uncomment the line below to execute the command
        eval "$command"
    done
done