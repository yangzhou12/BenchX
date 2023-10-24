import os
import numpy as np
import json
import torch.nn.functional as F
import torch
import logging
from sklearn.metrics import classification_report, roc_auc_score
from . import *


logging.setLoggerClass(logging.Logger)


def compute_scores(metrics, refs, hyps, split, seed, config, epoch, logger, dump=True):
    scores = dict()
    # If metric is None or empty list
    if metrics is None or not metrics:
        return scores

    assert (
        refs is not None and hyps is not None
    ), "You specified metrics but your evaluation does not return hyps nor refs"

    assert len(refs) == len(
        hyps
    ), "refs and hyps must have same length : {} vs {}".format(len(refs), len(hyps))

    # Dump
    if dump:
        ckpt_dir = config.ckpt_dir
        base = os.path.join(ckpt_dir, "{}_{}_{}".format(split, seed, "{}"))
        refs_file = base.format("refs.txt")
        hyps_file = base.format("hyps.txt")
        metrics_file = base.format("metrics.txt")

        with open(refs_file, "w") as f:
            f.write("\n".join(map(str, refs)))
            f.close()

        with open(hyps_file, "w") as f:
            f.write("\n".join(map(str, hyps)))
            f.close()

    for metric in metrics:
        # metric_args = dict()
        #
        # # if metric has arguments
        # if OmegaConf.is_dict(metric):
        #     if len(metric) != 1:
        #         logger.warning("Metric badly formatted: {}. Skipping.".format(metric))
        #         continue
        #     metric_args = metric[list(metric.keys())[0]]
        #     metric = list(metric.keys())[0]

        # Iterating over metrics
        if metric == "accuracy":
            scores["accuracy"] = round(
                np.mean(np.argmax(refs, axis=-1) == np.argmax(hyps, axis=-1)) * 100,
                2,  # change from array to argmax for refs
            )
        elif metric == "vqa_score":
            hyps_tensor = torch.from_numpy(hyps)
            refs_tensor = torch.from_numpy(refs)
            logits = torch.max(hyps_tensor, 1)[1]
            one_hots = torch.zeros(*refs_tensor.size()).to(refs_tensor)
            one_hots.scatter_(1, logits.view(-1, 1), 1)
            score = one_hots * refs_tensor
            scores["vqa_score"] = (score.sum() / len(score)).item()
        elif metric == "f1-score":
            scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
        elif metric == "multiclass_auroc":
            scores["multiclass_auroc"] = roc_auc_score(
                refs,
                F.softmax(torch.from_numpy(hyps), dim=-1).numpy(),
                multi_class="ovr",
            )
        elif metric == "multilabel_auroc":
            AUROCs = []
            n_classes = refs.shape(1)
            gt_np = refs.numpy()
            pred_np = hyps.numpy()
            for i in range(n_classes):
                AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            scores["class_auroc"] = AUROCs
            scores["multilabel_auroc"] = sum(AUROCs) / n_classes
        else:
            logger.warning("Metric not implemented: {}".format(metric))

    if dump:
        with open(metrics_file, "a+") as f:
            f.write(
                json.dumps(
                    {"split": split, "epoch": epoch, "scores": scores},
                    indent=4,
                    sort_keys=False,
                )
            )
    return scores


if __name__ == "__main__":
    pass
