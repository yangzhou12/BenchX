#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Define ranges for parameters
batch_sizes=(32 64 128)
learn_rates=(1e-5 5e-5 1e-4 5e-4)

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 dataset method"
    exit 1
fi

dataset="$1"
method="$2"
gpu="$3"

# dataset="VinDr" # ["NIH", "VinDr", "CheXpert", "COVIDx", "PadChest"]
# method="convirt" # ["convirt", "gloria", "medclip_rn50", "medclip_vit", "medklip", "mflag", "mgca_rn50"]

# Loop through parameter combinations
for bs in "${batch_sizes[@]}"; do
    for lr in "${learn_rates[@]}"; do
        # Construct command with updated parameters
        command="CUDA_VISIBLE_DEVICES=$gpu python bin/train.py config/classification/$dataset/$method.yml \
                trainer.batch_size=$bs trainer.optim_params.lr=$lr \
                dataset.split=train_10 \
                pretrained_source=custom \
                ckpt_dir=experiments/grid_search/Our_$dataset/"
        
        # Print the command (optional)
        echo "Running command: $command"
        
        # Uncomment the line below to execute the command
        eval "$command"
    done
done
