import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from unifier.blocks.vision import *
from unifier.blocks.classifier import *
from unifier.blocks.classifier.evaluation import evaluation
from unifier.blocks.losses import *
from unifier.models.utils import get_n_params

from unifier.blocks.custom.refers.transformer import REFERSViT
from torch.nn import Identity


class ImageClassifier(nn.Module):
    def __init__(self, cnn, classifier, loss, **kwargs):
        super(ImageClassifier, self).__init__()

        cnn_func = cnn.pop("proto")
        loss_func = loss.pop("proto")
        classifier_func = classifier.pop("proto")

        self.cnn = eval(cnn_func)(**cnn)

        self.classifier = eval(classifier_func)(**classifier)

        self.loss_func = eval(loss_func)(**loss).cuda()

        print(self.loss_func)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, labels=None, from_training=True, iteration=None, epoch=None, **kwargs):
        out = self.cnn(images.cuda()) # CNN - dim 2

        if isinstance(self.cnn, VisualEncoder) and out.dim() == 3: # ViT
            if hasattr(self.cnn.model, "num_prefix_tokens"):
                out = out[:, self.cnn.model.num_prefix_tokens:].mean(dim=1) # avg global pooling
            else:
                out = out[:, 0] # class token (Timm implementation)

        out = self.classifier(out)

        loss = torch.tensor(0.0)
        if from_training:
            loss = self.loss_func(out, labels.cuda().float(), **kwargs)

        return {"loss": loss, "output": out, "pred": torch.sigmoid(out)}

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s
