CUDA_LAUNCH_BLOCKING=1 python finetuned_segmentation.py \
    --pretrain_path /home/faith/projects/unified-framework/checkpoints/chexpert_resnet50.ckpt \
    --output_dir /home/faith/projects/unified-framework/experiments \
    --model_name gloria \
    --scheduler cosine \
    --optimizer adam \
    --data_pct 0.01 \
    --gpu 0