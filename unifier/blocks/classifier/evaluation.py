import torch
import itertools
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

def evaluation(models, config, dl, **kwargs):
    logits = np.array([])
    labels = np.array([])
    losses = np.array([])
    cumulative_index = 0

    for num_batch, batch in enumerate(tqdm(dl, total=len(dl))):
        label = batch["labels"] # always takes one hot labels?

        batch_size, num_classes = label.size()  # batch_size, num_classes

        batch = {k: v.cuda() if (isinstance(v, torch.Tensor) and torch.cuda.is_available()) else v for k, v in batch.items()}
        with autocast():
            results = [model(**batch) for model in models]

        # Pre-allocate memory
        if num_batch == 0:
            logits = np.zeros((len(dl.dataset), len(models), num_classes))
            labels = np.zeros((len(dl.dataset), num_classes))
            losses = np.zeros((len(dl), len(models)))

        # Replace the loop with a single batch operation
        logits[cumulative_index:cumulative_index+batch_size] = np.stack([r["output"].data.cpu().numpy() for r in results], axis=1)

        # Batch GPU operations for labels (one-hot encoding)
        labels[cumulative_index:cumulative_index + batch_size] = label.numpy()

        # Loss
        losses[num_batch] = np.array([r["loss"].cpu().item() for r in results])

        cumulative_index += batch_size
    preds = np.mean(logits, axis=1)
    loss = np.mean(losses)

    return {
        **{
            "loss": loss,
            "refs": labels,
            "hyps": preds,
            "logits": logits,
        },  
    }
