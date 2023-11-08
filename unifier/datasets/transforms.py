import cv2
import torch
import numpy as np
from albumentations import ShiftScaleRotate, Normalize, Resize, Compose
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from albumentations.pytorch import ToTensorV2
from typing import List


class BaseTransforms(object):
    def __init__(self):
        pass

    data_transforms: List
    """ Composed list of transforms specified in __init__ method of custom transform class. """

    def __call__(self, image) -> torch.Tensor | np.ndarray:
        return self.data_transforms(image)

    def __str__(self) -> str:
        return str([tfm.__class__.__name__ for tfm in self.data_transforms.transforms])


class DataTransforms(BaseTransforms):
    """Standard transforms used for pretraining or downstream tasks."""

    def __init__(self, is_train: bool = True, resize: int = 224, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.Resize(resize),
                transforms.RandomCrop(crop_size),  # RandomResizedCrop
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),  # additional step for pre-processing
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet
            ]

        self.data_transforms = transforms.Compose(data_transforms)


class NIHTransforms(BaseTransforms):
    """Transforms used for finetuning on NIH dataset."""

    def __init__(self, is_train: bool = True, resize: int = 224, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978], std=[0.2449]),
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),  # additional step for pre-processing
                transforms.CenterCrop(crop_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978], std=[0.2449]),
            ]

        self.data_transforms = transforms.Compose(data_transforms)


class RSNATransforms(BaseTransforms):
    """Transforms used for finetuning on RSNA dataset."""

    def __init__(self, is_train: bool = True, resize: int = 224, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.Resize(resize),
                transforms.RandomCrop(crop_size),  # RandomResizedCrop
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4904], std=[0.248]),
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),  # additional step for pre-processing
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4904], std=[0.248]),
            ]

        self.data_transforms = transforms.Compose(data_transforms)


class CheXTransforms(BaseTransforms):
    """Transforms used for finetuning on CheXpert-v1.0-small dataset."""

    def __init__(self, is_train: bool = True, resize: int = 224, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.Resize(resize),
                transforms.RandomCrop(crop_size),  # RandomResizedCrop
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5330], std=[0.0349]),
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),  # additional step for pre-processing
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5330], std=[0.0349]),
            ]

        self.data_transforms = transforms.Compose(data_transforms)


class VQATransforms(BaseTransforms):
    """Transforms used for VQA."""

    def __init__(self, is_train: bool = True, size: int = 224):
        data_transforms = [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

        self.data_transforms = transforms.Compose(data_transforms)


class SegmentationTransforms(BaseTransforms):
    """Transforms used for segmentation."""

    def __init__(self, is_train: bool = True, crop_size: int = 224):
        data_transforms = []

        if is_train:
            data_transforms.append(
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                )
            )

        data_transforms.extend(
            [
                Resize(crop_size, crop_size),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1),
                ToTensorV2(),
            ]
        )

        self.data_transforms = Compose(data_transforms)
