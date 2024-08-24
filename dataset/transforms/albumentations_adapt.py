from albumentations.pytorch import ToTensorV2

from albumentations import (
    Affine, CLAHE,Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, ImageCompression, RandomGridShuffle,
    RandomBrightnessContrast, D4, OneOf, Compose, ToGray, MixUp
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        # D4(),
        GaussNoise(noise_scale_factor=0.1),
        ToGray(p=0.2),
        OneOf([
            MotionBlur(p=0.3),
            MedianBlur(blur_limit=3, p=0.2),
            Blur(blur_limit=3, p=0.2),
        ], p=0.3),
        Affine(translate_percent=0.5, rotate=(-180, 180), p=0.2),
        ImageCompression(quality_range=(75, 100), p=0.3),
        RandomGridShuffle(grid=(5, 5), p=0.5),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1)
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)
