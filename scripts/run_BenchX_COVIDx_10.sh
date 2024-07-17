#!/bin/bash

# Common parameters
DATASET_NAME="COVIDx" # ["NIH", "VinDr", "CheXpert", "COVIDx", "PadChest"]
COMMON_PARAMS="seed=42 dataset.split=train_10" # seed = [42, 0, 1]
SUFFIX="_10" # ["_1", "_10", ""]

# Array of models and their corresponding configurations
declare -a MODELS=(
    # "Our_ConVIRT convirt pretrained_source=custom"
    "GLoRIA gloria pretrained_source=official"
    "Our_GLoRIA gloria pretrained_source=custom"
    # "MedCLIP_R50 medclip_rn50 pretrained_source=official"
    # "MedCLIP_ViT medclip_vit pretrained_source=official"
    # "MedKLIP medklip pretrained_source=official"
    # "Our_MedKLIP medklip pretrained_source=custom"
    # "Our_MFLAG mflag pretrained_source=custom"
    # "MGCA_R50 mgca_rn50 pretrained_source=official"
    # "Our_MGCA_R50 mgca_rn50 pretrained_source=custom"
    # "MGCA_ViT mgca_vit pretrained_source=official"
    # "Our_MGCA_ViT mgca_vit pretrained_source=custom"
    "MRM mrm pretrained_source=official"
    "Our_MRM mrm pretrained_source=custom"
    "REFERS refers pretrained_source=official"
    "Our_REFERS refers pretrained_source=custom"
)

# Loop over models and print Python command with specified suffix
for model in "${MODELS[@]}"; do
    set -- $model
    MODEL_NAME=$1
    CONFIG_NAME=$2
    shift 2

    CONFIG_FILE="config/classification/$DATASET_NAME/$CONFIG_NAME.yml"

    # Print the Python command instead of executing it
    echo "python bin/train.py \"$CONFIG_FILE\" name=\"$MODEL_NAME${SUFFIX}\" $COMMON_PARAMS \"$@\""

    python bin/train.py "$CONFIG_FILE" name="$MODEL_NAME${SUFFIX}" $COMMON_PARAMS "$@"
done