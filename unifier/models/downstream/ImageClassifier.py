import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from unifier.blocks.vision import *
from unifier.blocks.classifier import *
from unifier.blocks.classifier.evaluation import evaluation
from unifier.blocks.losses import *
from unifier.models.utils import get_n_params


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

    def forward(self, images, labels=None, from_training=True, **kwargs):
        out = self.cnn(images.cuda())
        out = self.classifier(out)

        loss = torch.tensor(0.0)
        if from_training:
            loss = self.loss_func(out, labels.cuda(), **kwargs)

        return {"loss": loss, "output": out, "pred": torch.sigmoid(out)}

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s