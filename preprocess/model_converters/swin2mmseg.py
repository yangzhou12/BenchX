# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_swin(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    new_ckpt = OrderedDict()

    for key in list(ckpt.keys()):
        new_key = key.replace("embeddings.patch_embeddings", "patch_embed")
        new_key = new_key.replace("embeddings.norm", "patch_embed.norm")
        new_key = new_key.replace("encoder.layers", "stages")
        new_key = new_key.replace("layernorm.", "norm3.")
        new_key = new_key.replace("layernorm_before", "norm1")
        new_key = new_key.replace("layernorm_after", "norm2")
        new_key = new_key.replace("attention.self", "attn.w_msa")
        new_key = new_key.replace("attention.output.dense", "attn.w_msa.proj")
        new_key = new_key.replace("intermediate.dense", "ffn.layers.0.0")
        new_key = new_key.replace("output.dense", "ffn.layers.1")
        new_ckpt[new_key] = ckpt.pop(key)

    # q_dict, k_dict, v_dict = {}, {}, {}
    # for key in list(new_ckpt.keys()):
    #     if "query.weight" in key:
    #         q_dict["weight"] = new_ckpt.pop(key)
    #     if "key.weight" in key:
    #         k_dict["weight"] = new_ckpt.pop(key)
    #     if "value.weight" in key:
    #         v_dict["weight"] = new_ckpt.pop(key)
    #     if "query.bias" in key:
    #         q_dict["bias"] = new_ckpt.pop(key)
    #     if "key.bias" in key:
    #         k_dict["bias"] = new_ckpt.pop(key)
    #     if "value.bias" in key:
    #         v_dict["bias"] = new_ckpt.pop(key)
    #     if len(q_dict) + len(k_dict) + len(v_dict) == 6:
    #         prefix = key.split("value.bias")[0]
    #         weight = torch.cat((q_dict["weight"], k_dict["weight"], v_dict["weight"]), dim=0)
    #         bias = torch.cat((q_dict["bias"], k_dict["bias"], v_dict["bias"]), dim=0)
    #         new_ckpt[prefix + "qkv.weight"] = weight
    #         new_ckpt[prefix + "qkv.bias"] = bias
    #         q_dict, k_dict, v_dict = {}, {}, {}

    depths = [2, 2, 6, 2]
    for i in range(len(depths)):
        for j in range(depths[i]):
            prefix = f"stages.{i}.blocks.{j}.attn.w_msa."
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

            new_ckpt[prefix + "qkv.weight"] = weight
            new_ckpt[prefix + "qkv.bias"] = bias

    for k, v in new_ckpt.items():
        if 'downsample' in k:
            new_k = k
            if 'reduction.' in k:
                new_v = correct_unfold_reduction_order(v)
            elif 'norm.' in k:
                new_v = correct_unfold_norm_order(v)
            new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained swin models to'
        'MMSegmentation style.')

    parser.add_argument('src', default="/raid/zhouyang/unified-framework/ckpt/official/MedCLIP-vit.bin", nargs='?', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', default="/raid/zhouyang/processed_data/converted_ckpt/official/MedCLIP-vit.bin", nargs='?', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    prefix = "vision_model.model" # MedCLIP
    weight = convert_swin(state_dict, prefix)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
