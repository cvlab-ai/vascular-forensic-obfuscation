import albumentations as A
from torchvision import transforms
import cv2
import numpy as np


imgaug = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.3, border_mode=cv2.BORDER_REPLICATE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomSizedCrop(min_max_height=(400, 512), size=(512, 512), p=0.4),
    A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=1),
        ],p=0.4,
    )
])

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def apply_train_transforms(x, y):
    result = imgaug(image=np.array(x), mask=np.array(y))
    return image_transform(result['image']), mask_transform(result['mask'])

def apply_val_transforms(x, y):
    return image_transform(x), mask_transform(y)
    
def apply_test_transform(x):
    return mask_transform(x)