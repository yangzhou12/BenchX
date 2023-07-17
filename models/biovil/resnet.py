#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List, Type, Union
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
import torch

class ResNetHIML(ResNet):
    """ Taken from Biovil
    Wrapper class of the original torchvision ResNet model.
    The forward function is updated to return the penultimate layer
    activations, which are required to obtain image patch embeddings.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.block_ind=3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        if self.block_ind == 0:
            return x1

        x2 = self.layer2(x1)
        if self.block_ind == 1:
            return x2

        x3 = self.layer3(x2)
        if self.block_ind == 2:
            return x3

        x4 = self.layer4(x3)
        return x4


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
            pretrained: bool, progress: bool, **kwargs: Any) -> ResNetHIML:
    """Instantiate a custom :class:`ResNet` model.
    Adapted from :mod:`torchvision.models.resnet`.
    """
    model = ResNetHIML(block=block, layers=layers, **kwargs)
    if pretrained:
        if arch == 'resnet18':
            model.load_state_dict(ResNet18_Weights.DEFAULT.get_state_dict(progress=progress))
        elif arch == 'resnet50':
            model.load_state_dict(ResNet50_Weights.DEFAULT.get_state_dict(progress=progress))
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet.
    :param progress: If ``True``, displays a progress bar of the download to ``stderr``.
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet
    :param progress: If ``True``, displays a progress bar of the download to ``stderr``.
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)