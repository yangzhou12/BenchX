import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.0, use_fc_norm=False, trunc_init=False, **kwargs):
        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=input_size, out_features=num_classes)
        # )
        # self.dropout = nn.Dropout(p=dropout)

        self.fc_norm = nn.LayerNorm(input_size, eps=1e-6) if use_fc_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features=input_size, out_features=num_classes) if num_classes > 0 else nn.Identity()
        if trunc_init: nn.init.trunc_normal_(self.classifier.weight, std=2e-5) 

    def forward(self, input: torch.Tensor):
        output = self.fc_norm(input)
        output = self.dropout(output)
        # output = self.dropout(input)
        output = self.classifier(output)
        return output
