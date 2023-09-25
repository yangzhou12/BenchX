CUDA_VISIBLE_DEVICES=1 python -m train.train_radrestruct \
    --pretrain_path /home/faith/unified-framework/checkpoints/MGCA-resnet50.ckpt \
    --model_name mgca-resnet50 \
    --base_model resnet50 \
    --run_name train_radrestruct \
    --classifier_dropout 0.2 \
    --acc_grad_batches 2 \
    --lr 1e-5 \
    --epochs 200 \
    --batch_size 32 \
    --progressive \
    --mixed_precision