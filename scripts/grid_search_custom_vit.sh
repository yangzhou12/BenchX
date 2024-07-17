#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Define ranges for parameters
batch_sizes=(32 64 128)
learn_rates=(3e-2 1e-2 3e-3 1e-3)

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 dataset method"
    exit 1
fi

dataset="$1"
method="$2"

# Loop through parameter combinations
for bs in "${batch_sizes[@]}"; do
    for lr in "${learn_rates[@]}"; do
        # Construct command with updated parameters
        command="CUDA_VISIBLE_DEVICES=4 python bin/train.py config/classification/$dataset/$method.yml \
                trainer.batch_size=$bs trainer.optim_params.lr=$lr \
                dataset.split=train_10 \
                pretrained_source=custom \
                ckpt_dir=experiments/grid_search/$dataset/"
        
        # Print the command (optional)
        echo "Running command: $command"
        
        # Uncomment the line below to execute the command
        eval "$command"
    done
done
