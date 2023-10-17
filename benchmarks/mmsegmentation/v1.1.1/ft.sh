bash tools/dist_train.sh \
    configs/mrm/upernet_mrm-base_fp16-160k_siim-512x512.py 1 \
    --work-dir ./output/\
    --cfg-options model.backbone.init_cfg.checkpoint=/home/faith/projects/unified-framework/checkpoints/MRM.pth

