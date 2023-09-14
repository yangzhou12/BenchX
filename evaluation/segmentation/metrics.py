# Code adapted from OpenMMLab
import torch
import numpy as np
from collections import OrderedDict


def f_score(precision, recall, beta=1):
    """Calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label, label, num_classes, ignore_index):  # 2
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
            or predict result filename.
        label (ndarray): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    pred_label = torch.from_numpy((pred_label))
    label = torch.from_numpy(label)

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
    )
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1
    )
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1
    )
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(
    results,  # len(results) = batch_size
    gt_seg_maps,  # len(gt_seg_maps) = batch_size
    num_classes,
    ignore_index,
):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): List of ground
            truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):  # for each sample
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_classes, ignore_index
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )


def total_area_to_metrics(
    total_area_intersect,
    total_area_union,
    total_area_pred_label,
    total_area_label,
    metrics=["mIoU"],
    nan_to_num=None,
    beta=1,
):
    """Calculate evaluation metrics.

    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({"aAcc": all_acc.item()})
    for metric in metrics:
        if metric == "mIoU":
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics["mIoU"] = iou.mean().item()
            ret_metrics["mAcc"] = acc.mean().item()
        elif metric == "mDice":
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics["mDice"] = dice.mean().item()
            ret_metrics["mAcc"] = acc.mean().item()
        elif metric == "mFscore":
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
            )
            ret_metrics["mFscore"] = f_value.mean().item()
            ret_metrics["mPrecision"] = precision.mean().item()
            ret_metrics["mRecall"] = recall.mean().item()

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {
                metric: np.nan_to_num(metric_value.numpy(), nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            }
        )
    return ret_metrics


def eval_metrics(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index=None,
    metrics=["mIoU"],
    nan_to_num=None,
    beta=1,
):
    """Calculate evaluation metrics.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): List of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        dict[str, float | ndarray]:
            <medDice> float: Average medical segmentation dice score.
            <aAcc> float: Overall accuracy on all images.
            <mIoU> ndarray: Per category IoU, averaged over num_classes.
            <mAcc> ndarray: Per category accuracy, averaged over num_classes.
            <mDice> ndarray: Per category natural dice score, averaged over num_classes.
            <mFscore> ndarray: Per category F-score, averaged over num_classes.
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category recall, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ["mIoU", "mDice", "mFscore", "medDice"]
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError("metrics {} is not supported".format(metrics))

    ret_metrics = {}

    if "medDice" in metrics:  # common dice score computation for medical VLMs
        batch_size = len(results)
        dice = 0
        dice_list = []
        for result, gt in zip(results, gt_seg_maps):
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
            dice_list.append(dice_instance)

        dice /= batch_size  # dice score averaged over batch
        ret_metrics["medDice"] = dice

    (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    ) = total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index)

    ret_metrics.update(
        total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            metrics,
            nan_to_num,
            beta,
        )
    )

    return ret_metrics


# GLoRIA/MGCA dice score computation
def get_dice(probability, truth, threshold=0.5):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)  # indices of neg samples
        pos_index = torch.nonzero(t_sum >= 1)  # indices of pos samples

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]  # dice scores for negative samples
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

    return torch.mean(dice).detach().item()


def test_metrics():
    pred_size = (10, 30, 30)  # [batch_size, height, width]
    num_classes = 2
    ignore_index = None
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)

    ret_metrics = eval_metrics(
        results, label, num_classes, ignore_index, metrics=["medDice"]
    )

    mrm_dice = ret_metrics
    gloria_dice = get_dice(torch.from_numpy(results), torch.from_numpy(label))

    print(mrm_dice["medDice"])
    print(gloria_dice)


if __name__ == "__main__":
    test_metrics()
