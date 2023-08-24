import torchvision.transforms as transforms


class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                #transforms.Resize(crop_size),
                transforms.RandomResizedCrop(crop_size), #RandomResizedCrop
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5)) # normalize in the range [-1, 1]
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)
