import torch.nn as nn
from torchvision import models as models_2d
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


######################################
#           ResNet Family            #
######################################

def resnet_18(pretrained=True):
    pretrained_weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models_2d.resnet18(weights=pretrained_weights)
    #feature_dims = model.fc.in_features
    model.fc = Identity()
    return model

def resnet_34(pretrained=True):
    pretrained_weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = models_2d.resnet34(weights=pretrained_weights)
    #feature_dims = model.fc.in_features
    model.fc = Identity()
    return model

def resnet_50(pretrained=True):
    pretrained_weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models_2d.resnet50(weights=pretrained_weights)
    #feature_dims = model.fc.in_features
    model.fc = Identity()
    return model