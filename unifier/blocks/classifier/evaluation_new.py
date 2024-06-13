import torch
import itertools
import numpy as np
from tqdm import tqdm


def evaluation(models, config, dl, **kwargs):
    logits, labels, losses = [], [], []
    attentions = None
    cumulative_index = 0

    for num_batch, batch in enumerate(tqdm(dl, total=len(dl))):
        label = batch["labels"]

        batch_size = label.shape[0]  # batch_size
        num_classes = config.model.classifier.get("num_classes", config.model.cnn.get("num_classes", None))

        batch = {
            k: v.cuda()
            if (isinstance(v, torch.Tensor) and torch.cuda.is_available())
            else v
            for k, v in batch.items()
        }
        results = [model(**batch) for model in models]

        bach_res = []
        for j, r in enumerate(results):
            bach_res.append(r["output"].unsqueeze(dim=1))
            # convert to one-hot labels
        bach_res = torch.cat(bach_res, dim=1)
        logits.append(bach_res)

        if bach_res.shape[-1] != num_classes:
            label = torch.nn.functional.one_hot(label)
        labels.append(label)

        # Loss
        bach_loss = []
        for j, r in enumerate(results):
            bach_loss.append(r["loss"].unsqueeze(dim=0))
        bach_loss = torch.cat(bach_loss, dim=0)
        losses.append(bach_loss)

    logits = torch.cat(logits, dim=0).data.cpu().numpy()
    labels = torch.cat(labels, dim=0).data.cpu().numpy()
    losses = torch.cat(losses, dim=0).data.cpu().numpy()

    preds = np.mean(logits, axis=1)
    loss = np.mean(losses)

    # if attentions:
    #     post_processing["attentions"] = np.array(attentions)

    return {
        **{
            "loss": loss,
            "refs": labels,
            "hyps": preds,
            "logits": logits,
        },  
        # **post_processing,
    }