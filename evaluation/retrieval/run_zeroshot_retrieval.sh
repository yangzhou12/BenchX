python zeroshot_retrieval.py \
    --dataset mimic_5x200 \
    --pretrain_path /home/faith/unified-framework/checkpoints/chexpert_resnet50.ckpt \
    --model_name gloria \
    --similarity_type both \
    --gpu 1 \
