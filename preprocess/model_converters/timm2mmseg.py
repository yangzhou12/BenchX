# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_mae(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('decoder'):
            continue
        if k.startswith('bert'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('norm1'):
            new_k = k.replace('norm1.', 'ln1.')
        elif k.startswith('norm2'):
            new_k = k.replace('norm2.', 'ln2.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('blocks'):
            if 'norm1' in k:
                new_k = k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = k.replace('norm2', 'ln2')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt

def convert_vit(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('decoder'):
            continue
        if k.startswith('bert'):
            continue
        if k.startswith('mask'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('norm1'):
            new_k = k.replace('norm1.', 'ln1.')
        elif k.startswith('norm2'):
            new_k = k.replace('norm2.', 'ln2.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        # elif k.startswith('blocks') or k.startswith('layers'):
        elif k.startswith('blocks'):
            if 'norm1' in k:
                new_k = k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = k.replace('norm2', 'ln2')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt

def convert_ptunifier(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("class_embedding"):
            new_k = k.replace("class_embedding", "cls_token")
        elif k.startswith( "positional_embedding"):
            new_k = k.replace("positional_embedding", "pos_embed")
        elif k.startswith("ln_post"):
            new_k = k.replace("ln_post", "ln1")
        elif k.startswith("conv1"):
            new_k = k.replace("conv1", "patch_embed.projection")
        elif k.startswith("transformer.resblocks"):
            if "ln_" in k:
                new_k = k.replace("ln_", "ln")
            elif "mlp.c_fc" in k:
                new_k = k.replace("mlp.c_fc", "ffn.layers.0.0")
            elif "mlp.c_proj" in k:
                new_k = k.replace("mlp.c_proj", "ffn.layers.1")
            elif 'attn' in k:
                new_k = k.replace('attn.', 'attn.attn.')
            else:
                new_k = k
            new_k = new_k.replace("transformer.resblocks", "layers")
        else:
            new_k = k
        new_ckpt[new_k] = v

    new_ckpt['cls_token'] = new_ckpt['cls_token'].unsqueeze(0).unsqueeze(0)
    new_ckpt['pos_embed'] = new_ckpt['pos_embed'].unsqueeze(0)
    return new_ckpt

def convert_refers(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('embeddings'):
            if "position_embeddings" in k:
                new_k = k.replace('embeddings.position_embeddings', 'pos_embed')
            elif "patch_embeddings" in k:
                new_k = k.replace("embeddings.patch_embeddings", "patch_embed.projection")
            elif "cls_token" in k:
                new_k = k.replace("embeddings.cls_token", "cls_token")
        elif k.startswith('encoder'):
            if "attention_norm" in k:
                new_k = k.replace("attention_norm.", "ln1.")
            elif "encoder_norm" in k:
                new_k = k.replace("encoder.encoder_norm.", "ln1.")
            elif "ffn_norm" in k:
                new_k = k.replace("ffn_norm.", "ln2.")
            # elif "attn.out" in k:
            #     new_k = k.replace("attn.out.", "attn.proj.")
            elif "attn.out" in k:
                new_k = k.replace("attn.out.", "attn.attn.out_proj.")
            elif "ffn.fc1" in k:
                new_k = k.replace("ffn.fc1", "ffn.layers.0.0")
            elif "ffn.fc2" in k:
                new_k = k.replace("ffn.fc2", "ffn.layers.1")
            else:
                new_k = k
            new_k = new_k.replace('encoder.layer.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    num_layers = 12
    for i in range(num_layers):
        prefix = f"layers.{i}.attn."
        key = prefix + "query.weight"
        q_weight = new_ckpt.pop(key)
        key = prefix + "key.weight" 
        k_weight = new_ckpt.pop(key)
        key = prefix + "value.weight"
        v_weight = new_ckpt.pop(key)

        key = prefix + "query.bias"
        q_bias = new_ckpt.pop(key)
        key = prefix + "key.bias"
        k_bias = new_ckpt.pop(key)
        key = prefix + "value.bias"
        v_bias = new_ckpt.pop(key)

        weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

        # new_ckpt[prefix + "qkv.weight"] = weight
        # new_ckpt[prefix + "qkv.bias"] = bias

        new_ckpt[prefix + "attn.in_proj_weight"] = weight
        new_ckpt[prefix + "attn.in_proj_bias"] = bias

    return new_ckpt

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    
    src_dir = "/mnt/weka/MedFM/"
    dst_dir = "/mnt/weka/MedFM/converted_ckpt"
    # model_name = "OpenEye_mask_oct.pth"
    model_name = "RETFound_cfp_weights.pth"
    # model_name = "vit_base_patch16_224.dino"

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

    # prefix = "" # MRM
    # prefix = "img_encoder_q.model" # MGCA-ViT
    weight = convert_vit(state_dict, prefix="")
    # weight = convert_mae(state_dict, prefix="")
    args.dst = osp.join(dst_dir, model_name)

    # prefix = "transformer" # Official version
    # prefix = "visual.vit.transformer" # our version
    # weight = convert_refers(state_dict, prefix)

    # prefix = "vision_encoder.visual" # PTUnifier
    # weight = convert_ptunifier(state_dict, prefix)
    
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
