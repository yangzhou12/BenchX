python zeroshot_classifier.py \
    --model_name gloria \
    --pretrain_path /home/faith/unified-framework/checkpoints/chexpert_resnet50.ckpt \
    --similarity_type both \
    --batch_size 1000 \
    --seed 300
