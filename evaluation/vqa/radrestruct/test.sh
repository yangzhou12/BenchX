CUDA_VISIBLE_DEVICES=0 python -m evaluation.evaluate_radrestruct \
    --run_name eval_radrestruct_original \
    --match_instances \
    --batch_size 1 \
    --use_pretrained \
    --model_dir /home/faith/Rad-ReStruct/checkpoints_radrestruct/train_radrestruct_original/epoch=44-F1/val=0.32.ckpt \
    --progressive \
    --mixed_precision