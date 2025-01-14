import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from unifier.blocks.vision import *
from unifier.blocks.classifier import *
from unifier.blocks.classifier.evaluation import evaluation
from unifier.blocks.losses import *
from unifier.models.utils import get_n_params

from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.models.bert_generation import BertGenerationConfig


class MVQA(nn.Module):
    def __init__(self, cnn, classifier, adapter, transformer, loss, **kwargs):
        super(MVQA, self).__init__()

        cnn_func = cnn.pop("proto")
        loss_func = loss.pop("proto")
        classifier_func = classifier.pop("proto")

        self.cnn = eval(cnn_func)(**cnn)
        self.adapter = nn.Sequential(
            nn.Linear(adapter.pop("input_size"), adapter.pop("output_size")),
            torch.nn.LayerNorm(transformer.hidden_size, eps=transformer.layer_norm_eps),
        )

        bert_conf = BertGenerationConfig(**transformer)
        self.transformer = BertEncoder(bert_conf)
        self.pooler = BertPooler(bert_conf)

        self.classifier = eval(classifier_func)(**classifier)

        self.loss_func = eval(loss_func)(**loss).cuda()

        print(self.loss_func)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, labels=None, from_training=True, iteration=None, epoch=None, **kwargs):
        out = self.cnn(images.cuda())
        out = self.adapter(out)
        out = self.transformer(out, output_attentions=True)

        attentions = (
            out.attentions
        )  # num_layers, batch_size, num_heads, sequence_length, sequence_length

        out = self.pooler(out.last_hidden_state)
        out = self.classifier(out)

        loss = torch.tensor(0.0)
        if from_training: 
            loss = self.loss_func(out, labels.cuda(), **kwargs) # out - [32, 498], labels - [32, 1]

        return {"loss": loss, "output": out, "answer": torch.argmax(out, dim=-1)}

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s
