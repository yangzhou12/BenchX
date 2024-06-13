import os
import numpy as np
import json
import torch.nn.functional as F
import torch
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

from unifier.blocks.scorers.pycocoevalcap.bleu.bleu import Bleu
from unifier.blocks.scorers.pycocoevalcap.meteor import Meteor
from unifier.blocks.scorers.pycocoevalcap.rouge import Rouge


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
        # Multi-class classification metrics
        if metric == "multiclass_accuracy":
            # scores["multiclass_accuracy"] = round(np.mean(np.argmax(refs, axis=-1) == np.argmax(hyps, axis=-1)) * 100, 2)
            # Calculate accuracy
            y_true = np.argmax(refs, axis=-1)
            y_pred = np.argmax(hyps, axis=-1)
            scores["multiclass_accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 2)
        elif metric == "multiclass_precision":
            # Calculate accuracy
            y_true = np.argmax(refs, axis=-1)
            y_pred = np.argmax(hyps, axis=-1)
            scores["multiclass_precision"] = round(precision_score(y_true, y_pred) * 100, 2)
        elif metric == "multiclass_recall":
            # Calculate accuracy
            y_true = np.argmax(refs, axis=-1)
            y_pred = np.argmax(hyps, axis=-1)
            scores["multiclass_recall"] = round(recall_score(y_true, y_pred) * 100, 2)
        elif metric == "multiclass_f1":
            # Calculate accuracy
            y_true = np.argmax(refs, axis=-1)
            y_pred = np.argmax(hyps, axis=-1)
            scores["multiclass_f1"] = round(f1_score(y_true, y_pred) * 100, 2)
        elif metric == "multiclass_auroc":
            scores["multiclass_auroc"] = roc_auc_score(refs, F.softmax(torch.from_numpy(hyps), dim=-1).numpy(), multi_class="ovr")

        # Multi-label classification metrics
        elif metric == "multilabel_accuracy":
            gt_np = refs
            pred_np = F.sigmoid(torch.from_numpy(hyps)).cpu().numpy()            
            precision, recall, thresholds = precision_recall_curve(gt_np.ravel(), pred_np.ravel())

            # Calculate F1 score for each threshold, handling division by zero
            f1_scores = np.zeros_like(thresholds)

            # Only calculate F1 score for thresholds where both precision and recall are nonzero
            nonzero_indices = (precision[:-1] + recall[:-1]) != 0
            f1_scores[nonzero_indices] = 2 * (precision[:-1][nonzero_indices] * recall[:-1][nonzero_indices]) / (precision[:-1][nonzero_indices] + recall[:-1][nonzero_indices])

            # Find the threshold that maximizes F1 score
            max_f1_thresh = thresholds[np.argmax(f1_scores)]

            y_pred = pred_np > max_f1_thresh
            scores["micro_precision"] = round(precision_score(gt_np, y_pred, average='micro', zero_division=1) * 100, 2)
            scores["micro_recall"] = round(recall_score(gt_np, y_pred, average='micro', zero_division=1) * 100, 2)
            scores["micro_f1"] = round(precision_score(gt_np, y_pred, average='micro', zero_division=1) * 100, 2)
            scores["macro_precision"] = round(precision_score(gt_np, y_pred, average='macro', zero_division=1) * 100, 2)
            scores["macro_recall"] = round(recall_score(gt_np, y_pred, average='macro', zero_division=1) * 100, 2)
            scores["macro_f1"] = round(precision_score(gt_np, y_pred, average='macro', zero_division=1) * 100, 2)
        elif metric == "multilabel_auroc":
            AUROCs = []
            n_classes = refs.shape[1]
            for i in range(n_classes):
                AUROCs.append(roc_auc_score(refs[:, i], F.sigmoid(torch.from_numpy(hyps[:, i])).numpy()))
            scores["class_auroc"] = AUROCs
            scores["multilabel_auroc"] = sum(AUROCs) / n_classes
        
        # VQA metrics
        elif metric == "vqa_score":
            hyps_tensor = torch.from_numpy(hyps)
            refs_tensor = torch.from_numpy(refs)
            logits = torch.max(hyps_tensor, 1)[1]
            one_hots = torch.zeros(*refs_tensor.size()).to(refs_tensor)
            one_hots.scatter_(1, logits.view(-1, 1), 1)
            score = one_hots * refs_tensor
            scores["vqa_score"] = (score.sum() / len(score)).item()
        
        # Report Generation metrics
        elif metric == "BLEU":
            score, _ = Bleu(4).compute_score(refs, hyps, verbose=0)
            scores["BLEU1"] = score[0]
            scores["BLEU2"] = score[1]
            scores["BLEU3"] = score[2]
            scores["BLEU4"] = score[3]
        elif metric == "METEOR":
            score, _ = Meteor().compute_score(refs, hyps)
            scores["METEOR"] = score
        elif metric in "ROUGEL":
            score, _ = Rouge().compute_score(refs, hyps)
            scores["ROUGEL"] = score

        # Segmentation metrics
        elif metric == "medDice":
            dice = 0
            for result, gt in zip(hyps, refs):
                p = result
                t = gt
                t_sum = t.sum()
                p_sum = p.sum()

                if t_sum == 0:
                    dice_instance = float(p_sum == 0)
                    dice += dice_instance
                else:
                    mask = t != 0
                    p_not0 = p[mask]
                    t_not0 = t[mask]
                    inter = (p_not0 == t_not0).sum() * 2
                    dice_instance = inter / (p_sum + t_sum)
                    dice += dice_instance
            dice /= len(refs)
            score["medDice"] = dice

        # Catch error: unknown metric
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