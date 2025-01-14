import torch
import copy
import os
import operator
import re
import json
import inspect

import numpy as np
import torch.nn as nn
from omegaconf import OmegaConf

from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from unifier.models import *
from unifier.datasets import *

import torch_optimizer
from torch.optim import *
from torch_optimizer import *
from torch.optim.lr_scheduler import *
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from timm.scheduler.cosine_lr import CosineLRScheduler
from unifier.blocks.schedulers import LinearWarmupCosineAnnealingLR, WarmupCosineScheduler
import sys


def get_eval_func(models):
    dummy = models[0]
    # if isinstance(dummy, nn.DataParallel):
    #     dummy = dummy.module
    assert hasattr(dummy, "eval_func")
    return dummy.eval_func


def create_optim_param_groups(config, model):
    optim_grouping = config.optim_params.pop("optim_groups")
    if optim_grouping == "heads":
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ]
        head_names = [
            "vqa_head",
        ]
        multi_modal_names = ["multi_modal"]

        lr = config.optim_params.lr
        wd = config.optim_params.pop("weight_decay")
        lr_multiplier_head = config.optim_params.pop("lr_multiplier_head")
        lr_multiplier_multi_modal = config.optim_params.pop("lr_multiplier_multi_modal")

        model_params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in multi_modal_names)
                ],
                "weight_decay": wd,
                "lr": config.optim_params.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in multi_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in multi_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_multiplier_head,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in multi_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_multiplier_head,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in multi_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_multiplier_multi_modal,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in multi_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_multiplier_multi_modal,
            },
        ]
    elif optim_grouping == "ve_only":
        lr = config.optim_params.lr
        lr_multiplier_ve = config.optim_params.pop("lr_multiplier_ve")
        ve_params = list(map(id, model.cnn.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())

        model_params = [{'params': model.cnn.parameters(), 'lr': config.optim_params.lr * lr_multiplier_ve},
                        {'params': ed_params, 'lr': config.optim_params.lr}]
    else:
        raise NotImplementedError(optim_grouping)

    return model_params


def create_optimizer(config, logger, model, state_dict=None):
    assert "lr" in config.optim_params
    config.optim_params.lr = float(config.optim_params.lr)

    if "betas" in config.optim_params:
        config.optim_params.betas = eval(config.optim_params.betas)  # change to tuple

    if hasattr(torch.optim, config.optimizer):
        optim = getattr(torch.optim, config.optimizer)
    elif hasattr(torch_optimizer, config.optimizer):
        optim = getattr(torch_optimizer, config.optimizer)
    else:
        raise NotImplementedError(config.optimizer)

    print(config.optim_params)

    # Initialize optimizer groups
    if config.optim_params.get("optim_groups", None):
        model_params = create_optim_param_groups(config, model)
    else:
        model_params = model.parameters()
        
    optimizer = optim(model_params, **config.optim_params)
    logger.settings("Optimizer {} created".format(type(optimizer).__name__))

    if state_dict is not None and "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
        logger.info("Optimizer state loaded")
    else:
        logger.info(optimizer)
    return optimizer


def create_model(config, dl, logger, from_training=True, state_dict=None):
    # Create model, give him dataloader also
    config = copy.deepcopy(config.model)
    model = eval(config.pop("proto"))(
        **config, dl=dl, logger=logger, from_training=from_training
    )
    logger.settings("Model {} created".format(type(model).__name__))

    if state_dict is not None:
        if "model" not in state_dict:
            logger.critical(
                'This checkpoint is not valid. Key "model" is missing from dict.'
            )
            sys.exit()
        params = {k.replace("module.", ""): v for k, v in state_dict["model"].items()}
        model.load_state_dict(params, strict=True)
        logger.info("Model state loaded")
    else:
        logger.info(model)

    model = model.cuda()

    # Removed: DDP recommended over DP for multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    return model


def create_data_loader(
    config, split, logger, called_by_validator=False, called_by_ensemblor=False
):
    dataset_config = copy.deepcopy(config.dataset)

    # Extract dataset name
    dataset_name = dataset_config.proto
    del dataset_config["proto"]

    num_workers = dataset_config.pop("num_workers", 0)
    dataset_config.split = split

    # Its important the dataset receive info if call from ensembler (test time):
    # split can be train with validation transformation
    dataset = eval("datasets." + dataset_name)(
        transform=eval("transforms." + config.transforms.type)(
            is_train=(split.startswith("train")), **config.transforms.get("options", {})
        ),
        # split=split,
        **OmegaConf.to_container(dataset_config)
    )

    if hasattr(dataset, "get_collate_fn"):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    # RandomSampler for train split, during training only
    if split.startswith("train") and not called_by_validator:
        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=config.batch_size,
            drop_last=config.get("drop_last", False),
        )
        logger.info("Using " + type(sampler.sampler).__name__)

    else:
        sampler = BatchSampler(
            SequentialSampler(dataset), batch_size=config.batch_size, drop_last=False
        )

    # Print dataset for training and ensemblor
    if not called_by_validator or called_by_ensemblor:
        logger.settings("DataLoader")
        logger.info(dataset)

    return MultiEpochsDataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=sampler,
        pin_memory=True
    )


def create_scaler(config, logger, state_dict=None):
    scaler = torch.cuda.amp.GradScaler(enabled=(config.get("use_amp", False)))
    logger.settings("Using scaler : {}".format(scaler.is_enabled()))
    if state_dict is not None and "scaler" in state_dict:
        scaler.load_state_dict(state_dict["scaler"])
        logger.info("Scaler state loaded")
    return scaler


def create_training_scheduler(config, optimizer, logger, state_dict=None):
    config = copy.deepcopy(config)
    training_scheduler = TrainingScheduler(
        lr_decay_func=config.lr_decay,
        optimizer=optimizer,
        early_stop_metric=config.early_stop_metric,
        early_stop_limit=config.early_stop,
        lr_decay_params=config.lr_decay_params,
    )
    logger.settings("Training scheduler created")
    if state_dict is not None and "training_scheduler" in state_dict:
        training_scheduler.load_state_dict(state_dict["training_scheduler"])
        logger.info("Training scheduler state loaded")
    else:
        logger.info(training_scheduler)
    return training_scheduler


class CheckpointSaver(object):
    def __init__(self, ckpt_dir, logger, seed, ckpt=None):
        self.ckpt_dir = ckpt_dir
        self.seed = seed
        self.logger = logger
        self.current_tag = None
        self.current_epoch = None

        if ckpt is not None:
            self.current_tag, self.current_epoch = self.extract_tag_and_step(ckpt)
            self.logger.settings(
                "Resuming checkpoint after epoch {} with tag {}.".format(
                    self.current_epoch + 1, self.current_tag
                )
            )

    def save(self, state_dict, tag, current_epoch):
        if self.current_tag is not None:
            old_ckpt = os.path.join(
                self.ckpt_dir,
                "{}_{}_{}.pth".format(self.current_tag, self.current_epoch, self.seed),
            )
            assert os.path.exists(old_ckpt), old_ckpt
            os.remove(old_ckpt)

        tag = np.round(tag, 6)
        path = os.path.join(
            self.ckpt_dir, "{}_{}_{}.pth".format(tag, current_epoch, self.seed)
        )
        torch.save(state_dict, path)
        self.logger.info("{} saved.".format(path))

        self.current_tag = tag
        self.current_epoch = current_epoch

    def extract_tag_and_step(self, ckpt):
        groups = re.match(".*/(.*?)_(.*?)_(.*?).pth", ckpt)
        return float(groups.group(1)), int(groups.group(2))


class TrainingScheduler(object):
    iter_step_scheduler = {
        "CyclicLR",
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
        "get_polynomial_decay_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "WarmupCosineScheduler",
        "CosineLRScheduler"
    }
    epoch_step_scheduler = {
        "LambdaLR",
        "MultiplicativeLR",
        "StepLR",
        "MultiStepLR",
        "ConstantLR",
        "LinearLR",
        "ExponentialLR",
        "ChainedScheduler",
        "SequentialLR",
        "CosineAnnealingLR",
        "LinearWarmupCosineAnnealingLR",
    }
    val_step_scheduler = {"ReduceLROnPlateau"}

    def __init__(
        self,
        lr_decay_func,
        optimizer,
        early_stop_metric,
        early_stop_limit,
        lr_decay_params,
    ):
        super().__init__()

        self.epoch = 0
        self.early_stop = 0
        self.early_stop_limit = early_stop_limit
        self.metric_comp_func = operator.gt
        self.mode = "max"
        self.current_best_metric = -float("inf")
        self.lr_decay_params = lr_decay_params
        self.early_stop_metric = early_stop_metric

        # 4info: You can decay_on_training_loss and have a early_stop_metric different than training loss
        self.decay_on_training_loss = (
            self.lr_decay_params.get("decay_on_training_loss", False)
        )

        if early_stop_metric in ["validation_loss", "training_loss"]:
            self.metric_comp_func = operator.lt
            self.mode = "min"
            self.current_best_metric = float("inf")

        self.scheduler_name = lr_decay_func
        if self.scheduler_name == "ReduceLROnPlateau":
            self.lr_decay_params["mode"] = self.mode

        def remove_unused_args(func, **kwargs):
            sig = [param.name for param in inspect.signature(func).parameters.values()]
            return {k: v for k, v in kwargs.items() if k in sig}

        self.lr_decay_params = remove_unused_args(
            eval(lr_decay_func), **self.lr_decay_params
        )
        self.scheduler = eval(lr_decay_func)(optimizer, **self.lr_decay_params)

    def iteration_step(self):
        if self.scheduler_name in TrainingScheduler.iter_step_scheduler:
            self.scheduler.step()

    def epoch_step(self):
        self.epoch = self.epoch + 1
        if self.scheduler_name in TrainingScheduler.epoch_step_scheduler:
            self.scheduler.step()

    def eval_step(self, decay_metric=None, early_stop_score=None):
        ret = {
            "done_training": False,
            "save_state": False,
        }

        # LR scheduler
        if decay_metric is not None:
            if self.scheduler_name in TrainingScheduler.val_step_scheduler:
                self.scheduler.step(decay_metric)

        # Early stop
        if early_stop_score is not None:
            if self.metric_comp_func(early_stop_score, self.current_best_metric):
                self.current_best_metric = early_stop_score
                self.early_stop = 0
                ret["save_state"] = True
            else:
                self.early_stop += 1
                if self.early_stop == self.early_stop_limit:
                    ret["done_training"] = True
        return ret

    def __repr__(self):
        s = "TrainingScheduler (\n"
        s += self.scheduler_name + "\n"
        s += (
            str(json.dumps(dict(self.lr_decay_params), indent=4, sort_keys=True)) + "\n"
        )
        s += "Early stopping" + "\n"
        s += "    {0}: {1}\n".format("early_stop_limit", self.early_stop_limit)
        s += "    {0}: {1}\n".format("metric_comp_func", self.metric_comp_func)
        s += "    {0}: {1}\n".format("mode", self.mode)
        s += "    {0}: {1}\n".format("current_best_metric", self.current_best_metric)
        s += "    {0}: {1}\n".format(
            "decay_on_training_loss", self.decay_on_training_loss
        )
        s += ")"
        return s

    def state_dict(self):
        training_sched = {
            key: value for key, value in self.__dict__.items() if key != "scheduler"
        }
        training_sched["scheduler"] = self.scheduler.state_dict()
        return training_sched

    def load_state_dict(self, state_dict):
        if "scheduler" in state_dict:  # Retro compatible with older checkpoint version
            scheduler = state_dict.pop("scheduler")
            self.__dict__.update(state_dict)
            self.scheduler.load_state_dict(scheduler)
