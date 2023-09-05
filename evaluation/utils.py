import os
import re
import random
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from models.builders import *
from datasets.classification_dataset import *


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
