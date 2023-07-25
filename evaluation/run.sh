python classification_finetuner.py \
    --pretrain_path /home/faith/projects/unified-framework/models/convirt/data/checkpoint_state.pth \
    --dataset nih_chest_xray \
    --model_name convirt \
    --optimizer sgd \
    --output_dir /home/faith/projects/unified-framework/data \
    --num_steps 3000 \
    --warmup_steps 50 \
    --data_pct 0.01 \
    --learning_rate 3e-3 \
    --weight_decay 0