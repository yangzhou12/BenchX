#!/bin/bash

train_splits=("_1" "_10" "")

# List of seeds
seeds=(42 0 1)

# Run the script with different seeds
for seed in "${seeds[@]}"; do
    for split in "${train_splits[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python bin/train.py config/classification/COVIDx/convirt.yml \
            name="convirt${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/convirt${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=1 python bin/train.py config/classification/COVIDx/gloria.yml \
            name="gloria${split}" pretrained_source=custom \             
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/gloria${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=2 python bin/train.py config/classification/COVIDx/medclip_rn50.yml \
            name="medclip_rn50${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/medclip_rn50${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=3 python bin/train.py config/classification/COVIDx/medclip_vit.yml \
            name="medclip_vit${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/medclip_vit${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=4 python bin/train.py config/classification/COVIDx/medklip.yml \
            name="medklip${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/medklip${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=5 python bin/train.py config/classification/COVIDx/mflag.yml \
            name="mflag${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/mflag${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=6 python bin/train.py config/classification/COVIDx/mgca_rn50.yml \
            name="mgca_rn50${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/mgca_rn50${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=7 python bin/train.py config/classification/COVIDx/mgca_vit.yml \
            name="mgca_vit${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/mgca_vit${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=6 python bin/train.py config/classification/COVIDx/mrm.yml \
            name="mrm${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/mrm${split}_${seed}.log" 2>&1 &

        CUDA_VISIBLE_DEVICES=7 python bin/train.py config/classification/COVIDx/refers.yml \
            name="refers${split}" pretrained_source=custom \
            "seed=${seed}" dataset.split="train${split}" > "./logs/COVIDx/refers${split}_${seed}.log" 2>&1 &
        wait
        echo "Complete training: seed=${seed} split=train${split}"
    done

done