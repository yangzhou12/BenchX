import os
import re
import yaml
import random
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler

from datasets.classification_dataset import *


_MODELS = {
    "gloria-resnet50": "gloria.img_encoder.model.",
    "biovil-resnet50": "encoder.encoder.",
    "convirt-resnet50": "img_encoder.model.",
    "mrm-vit": "",
    "mgca-resnet50": "img_encoder_q.model.",
    "mgca-vit": "img_encoder_q.model."
}


def available_models():
    """ Returns the names of available evaluation models """
    return list(_MODELS.keys())


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_experiment(args):
    if args.resume and args.resume > 0:
        fp = os.path.join(args.output_dir, "exp" + str(args.resume))
        if os.path.exists(fp):
            return fp
        else:
            raise Exception("Experiment doesn't exist, cannot resume exp " + fp)

    if not os.listdir(os.path.join(args.output_dir)):
        print("No models exist, creating directory")
        fp = os.path.join(args.output_dir, "exp1")
    else:
        all_files = os.listdir(os.path.join(args.output_dir))
        je_exps = [exp for exp in all_files if "exp" in exp]
        
        if je_exps:
            num = [int(re.search("\d+", exp).group(0)) for exp in je_exps]
            highest_ind = np.argmax(np.array(num))
            highest = num[highest_ind]
            highest = highest + 1  
        else:
            highest = 1
       
        fp = os.path.join(args.output_dir, "exp" + str(highest))
    
    return fp


def load_training_setup(args, exp_path, model):
    # Configure optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError("Optimizer not found")

    # Configure scheduler
    if args.scheduler == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            warmup_t=args.warmup_steps,
            t_initial=args.num_steps,
            warmup_prefix=True,
        )
    else:
        raise RuntimeError("Scheduler not found")

    if args.resume and args.resume > 0:
        # Resume training from checkpoint
        print(f"Resuming from {exp_path} checkpoint")
        checkpoint_path = os.path.join(exp_path, "checkpoint_state.pth")
        checkpoint_model = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_model["model"])
        optimizer.load_state_dict(checkpoint_model["optimizer"])
        lr_scheduler.load_state_dict(checkpoint_model["lr_scheduler"])
        global_step = checkpoint_model["step"]
    else:
        # Start from scratch
        global_step = 0

    return global_step, optimizer, lr_scheduler


def save_params(args, exp_path):
    del args.config
    with open(os.path.join(exp_path, 'commandline_args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)


def load_encoder_from_checkpoint(args, encoder):
    if args.pretrain_path:
        ckpt = torch.load(args.pretrain_path, map_location="cpu")

        for key in [
            "state_dict",
            "model",
        ]:  # resolve differences in saved checkpoints
            if key in ckpt:
                ckpt = ckpt[key]
            break

        ckpt_dict = {}
        for k, v in ckpt.items():
            if k.startswith(_MODELS[args.model_name]):
                beginning_index = len(_MODELS[args.model_name].split(".")) - 1
                k = ".".join(k.split(".")[beginning_index:])
                ckpt_dict[k] = v
            for layer_k in ["head.weight", "head.bias", "fc.weight", "fc.bias"]:
                if layer_k in ckpt_dict:
                    del ckpt_dict[layer_k]
        encoder.load_state_dict(ckpt_dict, strict=False)
        del ckpt

    else:
        print("No checkpoint detected; Train model from scratch!")
