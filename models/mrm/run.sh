CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main_pretrain.py \
    --num_workers 10 \
    --accum_iter 2 \
    --batch_size 32 \
    --model mrm \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --resume /home/faith/projects/MRM-pytorch/data/checkpoint-60.pth \
    --data_path /home/faith/datasets/mimic_512 \
    --output_dir /home/faith/projects/MRM-pytorch/data \
