import torchvision.transforms as transforms

class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
            ]
        else:
            data_transforms = [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)
    
class MRMTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978],std=[0.2449])
            ]
        else:
            data_transforms = [
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978],std=[0.2449])
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)