#!/bin/bash

# Common parameters
COMMON_PARAMS="COVIDx" # ["NIH", "VinDr", "CheXpert", "COVIDx", "SIIM", "RSNA"]
# R50_MODELS=("convirt" "gloria" "medclip_rn50" "medclip_vit" "medklip" "mflag" "mgca_rn50" "mgca_vit" "mrm" "refers")
# R50_MODELS=("convirt" "gloria" "medclip_rn50")
# gpu="3"
# R50_MODELS=("medclip_vit" "medklip" "mflag")
# gpu="4"
R50_MODELS=("mgca_rn50" "mgca_vit" "mrm" "refers")
gpu="6"

# Loop over R50 models and run Python code with grid_search_r50.sh
for model in "${R50_MODELS[@]}"; do
    bash scripts/grid_search_custom_r50.sh "$COMMON_PARAMS" "$model" "$gpu"
done