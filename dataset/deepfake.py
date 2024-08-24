import itertools
import json
import os
import imageio
import numpy as np
import torch
from torchvision.datasets.folder import ImageFolder
import torch.utils.data as data
from  monai.data.dataset import Dataset
import random

from dataset.transforms.albumentations_adapt import strong_aug


class DeepFakeImageDataset(Dataset):
    
    def __init__(self, data_root, data_file, transform=None, with_noise_label=False, noise_ratio=0.2, strong_aug=strong_aug(0.6)):
        # with open(data_file) as f:
        #     data = json.load(f)
        self.R = random.Random(0)
        self.data_root = data_root
        # data = data
        results = []

        with open(data_file) as f:
            data = f.readlines()
        data = [i.strip() for i in data]
        
        for d in data:
            d_list = d.split(",")
            if len(d_list) > 1:
                img_name, label = d_list
            else:
                img_name = d_list[0]
                label = 0
            try:
                label = int(label)
                results.append(
                    {
                        "img": os.path.join(self.data_root, img_name), 
                        "label": label, 
                        "filename": img_name
                    }
                )
            except:
                pass
            
        self.data = results
        self.strong_aug = strong_aug
        super().__init__(results, transform)
        
    
    def __getitem__(self, index):
        data = data_info = self.data[index]

        if self.transform is not None:
            data = self.transform(data_info)
        if self.strong_aug is not None:
            # print(self.strong_aug(imgae=data['img']))
            # data_info["img"] = imageio.v3.imread(data_info["img"])
            # print(data['img'].numpy().transpose(2, 1, 0).shape)
            data['img'] = self.strong_aug(image=(data['img'] * 255).numpy().astype(np.uint8).transpose(1, 2, 0))["image"]
            # print(data['img'].shape, type(data['img']))
            data['img'] = torch.tensor(data['img'], dtype=torch.float32).permute(2, 0, 1) / 255
            
            
        return data
    
    def __len__(self):
        return len(self.data)
    

from typing import Any, Callable, Dict, Optional, Type

import torch
from pytorchvideo.data.clip_sampling import ClipSampler

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset, labeled_video_dataset

class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

def deepfake_video(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    iteration_dataset=False
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for the Kinetics dataset.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """

    # torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Kinetics")
    if not iteration_dataset:
        return LimitDataset(labeled_video_dataset(
            data_path,
            clip_sampler,
            video_sampler,
            transform,
            video_path_prefix,
            decode_audio,
            decoder,
        ))
    return labeled_video_dataset(
            data_path,
            clip_sampler,
            video_sampler,
            transform,
            video_path_prefix,
            decode_audio,
            decoder,
        )