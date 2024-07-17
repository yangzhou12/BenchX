# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader

def convert_resnet(state_dict, prefix):
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    return state_dict

def convert_medklip(state_dict, prefix):
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("0"):
            new_k = k.replace("0.", "conv1.")
        elif k.startswith( "1"):
            new_k = k.replace("1.", "bn1.")
        elif k.startswith("4"):
            new_k = k.replace("4.", "layer1.")
        elif k.startswith("5"):
            new_k = k.replace("5.", "layer2.")
        elif k.startswith("6"):
            new_k = k.replace("6.", "layer3.")
        else:
            new_k = k
        new_ckpt[new_k] = v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    
    src_dir = "/raid/zhouyang/unified-framework/ckpt/official/"
    dst_dir = "/raid/zhouyang/processed_data/converted_ckpt/official/"
    # model_name = "GLoRIA.ckpt"
    # model_name = "GLoRIA_last.ckpt"
    # model_name = "MedCLIP-resnet50.bin"
    # model_name = "MGCA-resnet50.ckpt"
    model_name = "total_prior.ckpt"
    # model_name = "MedKLIP.pth"
    # model_name = "biovil_image_resnet50_proj_size_128.pt"

    # src_dir = "/raid/zhouyang/unified-framework/ckpt/pretrained/"
    # dst_dir = "/raid/zhouyang/processed_data/converted_ckpt/pretrained/"
    # # model_name = "ConVIRT.pth"
    # # model_name = "GLoRIA.pth"
    # # model_name = "GLoRIA_last.ckpt" # Orig Version
    # # model_name = "M-FLAG_total.pth"
    # model_name = "MGCA-resnet50.ckpt"
    # # model_name = "MedKLIP.pth"

    parser.add_argument('src', default=src_dir+model_name, nargs='?', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', default=dst_dir+model_name, nargs='?', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # prefix = "visual.model" # ConVIRT
    # prefix = "gloria.img_encoder.model" # GLoRIA
    # prefix = "img_encoder_q.model" # MGCA
    # prefix = "encoder.encoder" # BioVIL
    # prefix = "vision_model.model" # MedCLIP Official
    # prefix = "img_encoder_q.model" # MedCLIP ResNet
    # prefix = "encoder" # M-FLAG
    prefix = "image_encoder.encoder" # PRIOR
    weight = convert_resnet(state_dict, prefix=prefix)
        
    # prefix = "module.res_features" # MedKLIP
    # weight = convert_medklip(state_dict, prefix=prefix)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
