import cv2
from albumentations import ShiftScaleRotate, Normalize, Resize, Compose
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2


class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            # Consider use the following Normalization parameters
            # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            data_transforms = [
                # transforms.Resize(crop_size),
                transforms.RandomResizedCrop(crop_size),  # RandomResizedCrop
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # normalize in the range [-1, 1]
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),  # additional step for pre-processing
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class SegmentationTransforms(object):
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

    def __call__(self, image=None, mask=None):
        return self.data_transforms(image=image, mask=mask)
