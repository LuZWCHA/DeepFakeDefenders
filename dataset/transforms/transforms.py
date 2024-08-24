from monai.transforms import (
        EnsureChannelFirstd,
        Compose, 
        SpatialPadd, 
        RandZoomd, 
        LoadImaged, 
        RandFlipd, 
        Resized, 
        Rotate90d, 
        RandGaussianSmoothd, 
        RandAdjustContrastd, 
        RandRotated,
        MixUp,
         
        SpatialPad, 
        RandCoarseDropoutd,
        Lambdad,
        RandSimulateLowResolutiond,
        RepeatChanneld,
        RepeatChannel,
        NormalizeIntensityd,
        EnsureTyped
    )

rc = RepeatChannel(3)
# dinov2 input size
size = 518
# size = 512
train_transforms = Compose(
        [
            LoadImaged(keys=["img"]), 
            EnsureChannelFirstd(keys=["img"]),
            # RandRotated(keys=["img"], prob=0.5),
            RandFlipd(keys=["img"], prob=0.5),
            # RandGaussianSmoothd(keys=["img"], prob=0.2,),
            
            Resized(keys=["img"], spatial_size=[size, size]),
            SpatialPadd(keys=["img"], spatial_size=[size, size]),
            RandSimulateLowResolutiond(keys=["img"], prob=0.4),
            RandCoarseDropoutd(keys=["img"], spatial_size=[8, 8], holes=1, max_holes=4, max_spatial_size=[32, 32], prob=0.2),
            RandZoomd(["img"], min_zoom=0.6, max_zoom=1.1),
            NormalizeIntensityd(keys=["img"], channel_wise=True),
            EnsureTyped(keys=["img", "label"],)
        ]
    )

val_transforms = Compose(
        [
            LoadImaged(keys=["img"]), 
            EnsureChannelFirstd(keys=["img"]),
            
            
            # SpatialPadd(keys=["img"], spatial_size=[size, size]),
            Resized(keys=["img"], spatial_size=[size, size]),
            SpatialPadd(keys=["img"], spatial_size=[size, size]),
            
            NormalizeIntensityd(keys=["img"], channel_wise=True),
            Lambdad(keys=["img"], func=lambda x: x if x.shape[0] == 3 else rc(x)),
            EnsureTyped(keys=["img", "label"],)
        ]
    )