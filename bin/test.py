import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unifier.executors import Trainer, Validator
from utils import get_args, get, print_args, get_seed, extract_seed_from_ckpt
from logger import set_logger

import glob

def find_best_checkpoint(folder_path, seed):
    pattern = "*_*_{}.pth".format(seed)
    checkpoint_files = glob.glob(os.path.join(folder_path, pattern))

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {folder_path}.")

    # Sort checkpoint files based on the accuracy (assuming accuracy is the first part of the filename)
    sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: float(x.split("/")[-1].split("_")[0]), reverse=True)

    # Get the path of the best checkpoint (the one with the highest accuracy)
    best_checkpoint_path = sorted_checkpoint_files[0]

    return best_checkpoint_path


def main():
    # Get args and create seed
    config, override = get_args()
    if "seed" in config:
        seed = get_seed(config.seed)
    else:
        seed = get_seed()

    # Create checkpoint dir
    config.ckpt_dir = os.path.join(config.ckpt_dir, config.name)
    config.ckpt = find_best_checkpoint(config.ckpt_dir, seed)

    # If ckpt is specified, we continue training. Let's extract seed
    if config.get("ckpt"):
        config.ckpt = os.path.join(config.ckpt_dir, config.ckpt) if not os.path.exists(config.ckpt) else config.ckpt
        assert os.path.exists(config.ckpt), f"Path '{config.ckpt}' does not exist"
        seed = extract_seed_from_ckpt(config.ckpt)

    # Create logger according to seed
    set_logger(config.ckpt_dir, seed)

    # Nice print args
    print_args(config, ['trainer', 'validator'], seed, override)

    # Fetch args for training and validation
    train_config = get(config, 'trainer')
    val_config = get(config, 'validator')

    # Trainer
    trainer = Trainer(config=train_config,  # train_config is all args but the other executors args
                      seed=seed)

    # Evaluator
    evaluator = Validator(config=val_config,
                          models=[trainer.model],
                          train_dl=trainer.dl,
                          seed=seed,
                          from_training=True)

    # Lets be gentle, give evaluator to trainer
    trainer.evaluator = evaluator

    # Boom
    trainer.evaluator.start()


if __name__ == "__main__":
    main()
