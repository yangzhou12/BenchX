import os
import json
import torch
import logging
from .utils import create_data_loader, get_eval_func
from unifier.blocks.scorers.scores import compute_scores
from omegaconf.listconfig import ListConfig


class InitValidator(object):
    def __init__(self, config, models, train_dl, seed, from_training):
        self.seed = seed
        self.config = config
        self.from_training = from_training
        self.train_dl = train_dl

        # Logger
        self.logger = logging.getLogger(str(seed))

        # Models
        self.models = models

        # Metrics
        self.metrics = config.metrics
        if self.metrics and not isinstance(self.metrics, (list, ListConfig)):
            self.metrics = [self.metrics]

        self.post_processing = config.get("post_processing")

        self.epoch = 0

        # Evaluation splits
        self.splits = [
            (
                split,
                create_data_loader(
                    self.config,
                    split,
                    self.logger,
                    called_by_validator=True,
                    called_by_ensemblor=not from_training,
                ),
            )
            for split in self.config.get("splits")
        ]


class Validator(InitValidator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        assert isinstance(self.models, list)
        self.scores = []
        self.models = [m.eval() for m in self.models]

        for split, dl in self.splits:
            self.logger.info(
                "Running split: {} by ensembling {} models. "
                "Using {}.".format(
                    split,
                    len(self.models),
                    type(dl.batch_sampler.sampler).__name__,
                )
            )

            eval_func = get_eval_func(self.models)
            with torch.no_grad():
                results = eval_func(
                    models=self.models,
                    config=self.config,
                    dl=dl,
                    from_training=self.from_training,
                )

            # model must return at least loss or (refs and hyps)
            assert type(results) is dict and any(
                key in results for key in ["loss", "refs", "hyps"]
            ), self.logger.error("Evaluation func does not return any evaluation keys")

            scores = dict()

            # Handle loss
            scores["validation_loss"] = float(results.pop("loss", 0.0))

            # Handle metrics
            metrics = compute_scores(
                metrics=self.metrics,
                refs=results.pop("refs", None),
                hyps=results.pop("hyps", None),
                split=split,
                seed=self.seed,
                config=self.config,
                epoch=self.epoch,
                logger=self.logger,
            )
            scores.update(metrics)

            # Dumping things for potential post processing
            # post_processing(
            #     post_processing=self.post_processing,
            #     results=results,
            #     split=split,
            #     seed=self.seed,
            #     ckpt_dir=self.config.ckpt_dir,
            #     epoch=self.epoch,
            #     dl=dl,
            # )

            # Logging scores
            self.logger.info(json.dumps(scores, indent=4, sort_keys=False))

            # Saving the metrics for current split
            self.scores.append(scores)
