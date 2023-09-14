bash tools/dist_train.sh \
    configs/mrm/upernet_mrm-base_512x512_160k_siim.py 1 \
    --work-dir ./output/ --seed 0  --deterministic \
    --options model.backbone.pretrained=/home/faith/projects/unified-framework/checkpoints/MRM.pth

