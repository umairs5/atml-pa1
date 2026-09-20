"""ImageNet-compatible transforms prescribed for PACS training and evaluation."""

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def training_transform(resize_size: int = 256, crop_size: int = 224):
    """Resize, randomly crop and flip source or target training images."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform(resize_size: int = 256, crop_size: int = 224):
    """Resize and centre crop images for deterministic evaluation."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
