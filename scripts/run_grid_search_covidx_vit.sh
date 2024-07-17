#!/bin/bash

# Common parameters
COMMON_PARAMS="COVIDx" # ["NIH", "VinDr", "CheXpert", "COVIDx", "SIIM", "RSNA"]
R50_MODELS=("convirt" "gloria" "medclip_rn50" "medklip" "mflag" "mgca_rn50" )
VIT_MODELS=("mgca_vit" "mrm" "refers")

# Loop over VIT models and run Python code with grid_search_vit.sh
for model in "${VIT_MODELS[@]}"; do
    bash scripts/grid_search_custom_vit.sh "$COMMON_PARAMS" "$model"
done
