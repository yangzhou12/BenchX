python classification_finetuner.py \
    --pretrain_path /home/faith/projects/unified-framework/checkpoints/biovil_image_resnet50_proj_size_128.pt \
    --dataset rsna_pneumonia \
    --model_name biovil \
    --output_dir /home/faith/projects/unified-framework/data \
    --num_epochs 30