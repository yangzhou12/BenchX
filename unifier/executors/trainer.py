import numpy as np
import logging
import torch
import tqdm
import sys
import os

from .validator import Validator
from .utils import (
    CheckpointSaver,
    create_model,
    create_data_loader,
    create_optimizer,
    create_training_scheduler,
    create_scaler,
)

# torch.backends.cudnn.benchmark = False # For reproducibility
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

class TrainerConfig(object):
    def __init__(self, config, seed):
        # Misc
        self.config = config
        self.seed = seed

        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # import random
        # random.seed(0)

        self.state = None
        self.ckpt_dir = self.config.ckpt_dir
        self.ckpt = self.config.get("ckpt")

        # Training
        self.eval_start = self.config.get("eval_start", 0)
        self.eval_interval = self.config.get("eval_interval", 1)
        self.decay_metric_start = self.config.get("decay_metric_start", 0)
        self.early_stop_start = self.config.get("early_stop_start", 0)
        self.grad_accu = self.config.get("grad_accu", 1)
        self.clip_grad_norm = self.config.get("clip_grad_norm")

        # Do we resume training?
        if config.get("ckpt") is not None:
            self.state = torch.load(config.ckpt)

        # Logger
        self.logger = logging.getLogger(str(self.seed))

        # Checkpoints
        self.saver = CheckpointSaver(
            ckpt_dir=self.ckpt_dir, logger=self.logger, seed=self.seed, ckpt=self.ckpt
        )

        # Dataloader
        self.dl = create_data_loader(self.config, split=self.config.dataset.get("split", "train"), logger=self.logger)

        # Model
        self.model = create_model(
            self.config,
            dl=self.dl,
            logger=self.logger,
            from_training=True,
            state_dict=self.state,
        )

        # Optimizer
        self.optimizer = create_optimizer(
            config=self.config,
            logger=self.logger,
            model=self.model,
            state_dict=self.state,
        )

        # Lr Scheduler and early stop
        self.training_scheduler = create_training_scheduler(
            config=self.config,
            optimizer=self.optimizer,
            logger=self.logger,
            state_dict=self.state,
        )

        # Scaler
        self.scaler = create_scaler(
            config=self.config, logger=self.logger, state_dict=self.state
        )

        # Validator is None at init
        self.evaluator: Validator = None


class Trainer(TrainerConfig):
    def __init__(self, config, seed):
        super().__init__(config=config, seed=seed)

    def start(self):
        for epoch in range(int(self.training_scheduler.epoch), self.config.epochs + 1):
            self.model.train()

            losses = []
            log = ""
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))

            # Training
            for iteration, batch in enumerate(pbar, start=1):
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    out = self.model(**batch, epoch=epoch, iteration=iteration)

                # If the model is taking care of it own training
                if "loss" not in out:
                    pbar.set_description("Epoch {}, {}".format(epoch + 1, out))
                    continue

                loss = out["loss"]
                # if isinstance(self.model, torch.nn.DataParallel):
                #     loss = loss.mean()
                self.scaler.scale(loss).backward()

                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.clip_grad_norm
                    )

                losses.append(loss.item())

                if iteration % self.grad_accu == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.training_scheduler.iteration_step()

                    lr_lst = [
                        f"{param_group['lr']:.1e}"
                        for param_group in self.optimizer.param_groups
                    ]

                    log = "Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {} {}".format(
                        epoch + 1,
                        lr_lst if len(lr_lst) <= 2 else (lr_lst[:2] + ["..."]),
                        sum(losses) / iteration,
                        self.training_scheduler.early_stop_metric,
                        self.training_scheduler.current_best_metric,
                        self.training_scheduler.early_stop,
                        out["custom_print"] if "custom_print" in out else "",
                    )
                    pbar.set_description(log)

                # break
            # Perform last update if needed
            if (iteration % self.grad_accu != 0) and ("loss" in out):
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.clip_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.step()
                self.training_scheduler.iteration_step()
                self.optimizer.zero_grad()

            # End of epoch
            self.logger.info(log)
            self.training_scheduler.epoch_step()

            # Evaluation starts
            early_stop_score = None
            decay_metric = None
            do_earl_stop = epoch + 1 >= self.early_stop_start
            do_lr_decay = epoch + 1 >= self.decay_metric_start
            do_eval = (epoch + 1 >= self.eval_start) and ((epoch + 1 - self.eval_start) % self.eval_interval == 0)
            training_loss = sum(losses) / iteration

            # Compute early_stop_score according to early_stop_metric if specified
            if self.config.get("early_stop_metric") == "training_loss" and do_earl_stop:
                early_stop_score = training_loss

            # Do eval ?
            if do_eval:
                self.evaluator.epoch = epoch
                self.evaluator.start()
                # Compute early stop according to evaluation metric if specified
                if self.config.get("early_stop_metric") != "training_loss" and do_earl_stop:
                    early_stop_score = np.mean(
                        [
                            s[self.config.early_stop_metric]
                            for s in self.evaluator.scores
                        ]
                    )

            # Record decay_metric (will not be used further if scheduler != ReduceLROnPlateau)
            if do_lr_decay:
                if self.training_scheduler.decay_on_training_loss:
                    decay_metric = training_loss
                else:
                    decay_metric = early_stop_score

            ret = self.training_scheduler.eval_step(
                decay_metric=decay_metric, early_stop_score=early_stop_score
            )

            if ret["done_training"]:
                self.logger.info("Early stopped reached")
                self.logger.info("Best {}: {}".format(self.config.early_stop_metric, self.training_scheduler.current_best_metric))
                sys.exit()
            if ret["save_state"]:
                self.saver.save(
                    state_dict={
                        "model": self.model.state_dict(),
                        "training_scheduler": self.training_scheduler.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "config": self.config,
                        "scaler": self.scaler.state_dict(),
                    },
                    tag=early_stop_score,
                    current_epoch=epoch + 1,
                )
