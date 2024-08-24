import itertools
from operator import index
import os
import random
from turtle import forward
import pandas as pd
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Pad,
    CenterCrop,
    RandomHorizontalFlip
)

from dataset.deepfake import deepfake_video
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorchvideo.models.resnet
import torchmetrics

from models.av_vit import get_deepfake_av_model



class MyResult():
    
    def __init__(self, ):
        self.result_dict = dict()
        self.result_dict = {
            "pred": [],
            "gt": [],
            "id": []
        }
        
    
    def __call__(self, pred, gt, pred_id):
        pred = F.softmax(pred, dim=-1)
        for p, g, i in zip(pred, gt, pred_id):
            self.result_dict["pred"].append(p[1].item())
            self.result_dict["gt"].append(g.item())
            self.result_dict["id"].append(i)
        
    
    def reset(self):
        # Reset
        self.result_dict = {
            "pred": [],
            "gt": [],
            "id": []
        }
    
    def aggregate(self, agg_method="max", save_res="merged_av_res.csv"):
        pd.DataFrame(self.result_dict).to_csv(save_res)
        # Aggregate id
        res_by_id = dict()
        final_res = {
            "pred": [],
            "gt": [],
            "id": []
        }
        for p, g, i in zip(self.result_dict["pred"], self.result_dict["gt"], self.result_dict["id"]):
            if i not in res_by_id:
                res_by_id[i] = {"pred": [], "gt": g}
                # final_res["id"].append(i)
                # final_res["pred"]
                # final_res["gt"].append(g)
            
            res_by_id[i]["pred"].append(p)
            
        
        for k, v in res_by_id.items():
            final_res["pred"].append(max(v["pred"]) if agg_method == "max" else sum(v["pred"]) / len(v["pred"]))
            final_res["gt"].append(v["gt"])
            final_res["id"].append(k)
        pd.DataFrame(final_res).to_csv("_"+save_res)
        return final_res

class ApplyTransformToKeyWithDefault:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform, default=None):
        self._key = key
        self._transform = transform
        self.default = default

    def __call__(self, x):
        if self._key in x:
            x[self._key] = self._transform(x[self._key])
            # print(x[self._key].shape)
            if self.default is None:
                self.default = torch.zeros_like(x[self._key])
        else:
            x[self._key] = self.default
        return x


class DeepFakeDataModule(pytorch_lightning.LightningDataModule):

    # Dataset configuration
    _DATA_PATH = "/nasdata2/private/lzhao/workspace/kaggle/DeepfakeVideo/dataset/video/phase1"
    # IMAGE_DATA_PATH = "/nasdata2/private/lzhao/workspace/kaggle/DeepfakeVideo/dataset/video/phase1"
    testset_root = "/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/video/phase2"
    
    _CLIP_DURATION = 4  # Duration of sampled clip for each video
    _BATCH_SIZE = 24 # 2 A100
    _BATCH_SIZE = 24 # 8 RTX3090
    _NUM_WORKERS = 12  # Number of parallel processes fetching data

    # def train_dataloader(self):
    #     """
    #     Create the Kinetics train partition from the list of video labels
    #     in {self._DATA_PATH}/train
    #     """
    #     train_dataset = pytorchvideo.data.Kinetics(
    #         data_path=os.path.join(self._DATA_PATH, "train"),
    #         clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
    #         decode_audio=False,
    #     )
    #     return torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=self._BATCH_SIZE,
    #         num_workers=self._NUM_WORKERS,
    #     )

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.trainer.use_ddp:
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
            [
                self._video_transform(mode="val"),
                self._audio_transform(mode="val")
            ]
        )
        
        val_dataset = deepfake_video(
            data_path=os.path.join(self._DATA_PATH, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=val_transform,
            video_path_prefix=os.path.join(self._DATA_PATH, "valset"),
            iteration_dataset=True
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def _video_transform(self, mode: str):
        """
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
        # args = self.args
        video_num_subsampled = 20
        if mode == "val":
            video_num_subsampled -= 4
        video_means, video_stds = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
        video_crop_size = 256
        video_horizontal_flip_p = 0.5
        video_min_short_side_scale, video_max_short_side_scale = 256, 320
        
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(video_num_subsampled),
                    Normalize(video_means, video_stds),
                ]
                + (
                    [
                        RandomShortSideScale(
                            min_size=video_min_short_side_scale,
                            max_size=video_max_short_side_scale,
                        ),
                        RandomCrop(video_crop_size),
                        RandomHorizontalFlip(p=video_horizontal_flip_p),
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(video_min_short_side_scale),
                        CenterCrop(video_crop_size),
                    ]
                )
            ),
        )
        
        # return ApplyTransformToKey(
        #             key="video",
        #             transform=Compose(
        #                 [
        #                     UniformTemporalSubsample(16),
        #                     Lambda(lambda x: x / 255.0),
        #                     Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        #                     ShortSideScale(size=256),
        #                 ]
        #             ),
        #         ),

    def _audio_transform(self, mode="train"):
        """
        This function contains example transforms using both PyTorchVideo and TorchAudio
        in the same Callable.
        """
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train.csv. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        audio_raw_sample_rate = 44_100
        audio_resampled_rate = 16_000
        audio_mel_window_size = 32
        audio_mel_step_size = 16
        audio_logmel_mean = -7.03
        audio_logmel_std = 4.66
        audio_frame_num = 20
        if mode == "val":
            audio_frame_num -= 4
        audio_mel_num_subsample = audio_frame_num * 128
        audio_num_mels = 128
        
        n_fft = int(
            float(audio_resampled_rate) / 1000 * audio_mel_window_size
        )
        hop_length = int(
            float(audio_resampled_rate) / 1000 * audio_mel_step_size
        )
        eps = 1e-10
        
        # args = self.args
        # n_fft = int(
        #     float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size
        # )
        # hop_length = int(
        #     float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size
        # )
        # eps = 1e-10
        return ApplyTransformToKeyWithDefault(
                    key="audio",
                    transform=Compose(
                        [
                            Resample(
                                orig_freq=audio_raw_sample_rate,
                                new_freq=audio_resampled_rate,
                            ),
                            MelSpectrogram(
                                sample_rate=audio_resampled_rate,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                n_mels=audio_num_mels,
                                center=False,
                            ),
                            Lambda(lambda x: x.clamp(min=eps)),
                            Lambda(torch.log),
                            UniformTemporalSubsample(audio_mel_num_subsample, temporal_dim=-1),
                            Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                            Lambda(
                                lambda x: x.view(1, audio_frame_num, x.size(0) // audio_frame_num, x.size(1))
                            ),  # (T, F) -> (1, T, 1, F)
                            # ShortSideScale(
                            #     size=224,
                            # ),
                            
                            Normalize((audio_logmel_mean,), (audio_logmel_std,)),
                        ]
                    ),
                )

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train.csv. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        
        train_transform = Compose(
            [
                self._video_transform(mode="train"),
                self._audio_transform(mode="train")
            ]
        )
        
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        train_dataset = deepfake_video(
            data_path=os.path.join(self._DATA_PATH, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION + 1),
            transform=train_transform,
            video_path_prefix=os.path.join(self._DATA_PATH, "trainset")
        )        
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True
        )

    def test_dataloader(self):
        """
        Create the Kinetics test partition from the list of video labels
        in {self._DATA_PATH}/test
        """
        val_transform = Compose(
            [
                self._video_transform(mode="val"),
                self._audio_transform(mode="val")
            ]
        )

        test_dataset = deepfake_video(
            data_path=os.path.join(self.testset_root, "testset1seen.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=val_transform,
            video_path_prefix=os.path.join(self.testset_root, "testset1seen"),
            iteration_dataset=True
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

class FusionModel(nn.Module):
    
    def __init__(self, models) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, inputs):
        
        all_res = []
        for m, i in zip(self.models, inputs):
            res = m(i)
            all_res.append(res)
            
        return sum(all_res)

def make_deepfake_vitb():
    # pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(input_channels=3, spatial_size=256, )
    return get_deepfake_av_model()
    

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.av_model = make_deepfake_vitb()
        
        def print_model_info_by_thop(model):
            try:
                from thop import clever_format
                from thop import profile
                input_video = torch.randn(1, 3, 16, 256, 256)
                input_audio = torch.randn(1, 1, 16, 128, 128)
                flops, params = profile(model, inputs=(input_video, input_audio), )
                flops, params = clever_format([flops, params], "%.3f")
                print("Model Info:", f"FLOPs is {flops}  Size of model is {params}")
            except:
                pass
            
        # print_model_info_by_thop(self.av_model)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=2)
        self.csv_res = MyResult()
        self.rng = random.Random(0)
        
    def forward(self, video, audio):
        return self.av_model(video, audio)

    def shuffle_batch(self, batch, prob=0.1):
        if self.rng.random() < prob:
            batch_size = batch["video"].shape[0]
            if batch_size < 2:
                pass
            else:
                # offset_idx = self.rng.choices(list(range(batch_size)), k=batch_size)
                # offset_idx += offset_idx
                # start = self.rng.randint(1, batch_size - 2)
                # offset_idx = offset_idx[start: start + batch_size]
                # batch["audio"][offset_idx[1]], batch["audio"][offset_idx[0]] = batch["audio"][offset_idx[0]], batch["audio"][offset_idx[1]]
                batch["audio"] = torch.roll(batch["audio"], shifts=1, dims=0)
                # Fake
                batch["label"] = torch.ones_like(batch["label"])
                
        return batch
    
    def get_offset_video_frames(self, batch):
        start = min(max(int(self.rng.gauss(2, 1) * 4), 0), 4)
        batch["video"] = batch["video"][:, :, start: start + 16]
        start = min(max(int(self.rng.gauss(2, 1) * 4), 0), 4)
        batch["audio"] = batch["audio"][:, :, start: start + 16]
        return batch
    
    def get_offset_video_frames_val(self, batch):
        start = 0
        batch["video"] = batch["video"][:, :, start: start + 16]
        start = 0
        batch["audio"] = batch["audio"][:, :, start: start + 16]
        return batch
    
    # def join_real_video(self, batch, prob):
    #     video_batch = batch["video"]
    #     audio_batch = batch["audio"]
    #     label = batch["label"]
        
    #     real_video = video[label == 0]
        
    #     pass

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        batch = self.get_offset_video_frames(batch)
        batch = self.shuffle_batch(batch, prob=0.3)
        batch_size = batch["video"].shape[0]
        y_hat = self.forward(batch["video"], batch["audio"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.get_offset_video_frames_val(batch)
        batch_size = batch["video"].shape[0]
        video_name = batch["video_name"]
        y_hat = self.forward(batch["video"], batch["audio"])
        loss = F.cross_entropy(y_hat, batch["label"])
        
        self.csv_res(y_hat, batch["label"], video_name)
        y_hat = F.softmax(y_hat, dim=-1)
        acc = self.val_accuracy(y_hat, batch["label"])
        auc = self.val_auc(y_hat, batch["label"])
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        self.log(
            "val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log(
            "val_auc", auc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        return  loss

    def on_validation_start(self) -> None:
        self.csv_res.reset()
        return super().on_validation_start()
    
    def on_validation_end(self) -> None:
        
        self.csv_res.aggregate(os.path.join(self.loggers[0].log_dir, f"merged_results_step_{self.current_epoch}-{self.global_step}.csv"))
        return super().on_validation_end()

    def test_step(self, batch, batch_idx):
        video_name = batch["video_name"]
        y_hat = self.forward(batch["video"], batch["audio"])
        fake_prob = y_hat
        self.csv_res(fake_prob, batch["label"], video_name)
    
    def on_test_start(self) -> None:
        self.csv_res.reset()
        return super().on_test_start()
    
    def on_test_end(self) -> None:
        self.csv_res.aggregate(os.path.join(f"merged_results_testset.csv"))
        return super().on_test_end()

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def train():
    import pytorch_lightning.loggers as pl_loggers
    name = "AV_ViTB"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=name)
    csv_logger = pl_loggers.CSVLogger(save_dir="logs/", name=name)
    # trainer = Trainer(logger=[tb_logger, comet_logger])
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=3, monitor="val_auc", mode="max")
    
    classification_module = VideoClassificationLightningModule()
    
    data_module = DeepFakeDataModule()
    trainer = pytorch_lightning.Trainer(callbacks=[checkpoint_callback], logger=[tb_logger, csv_logger], max_epochs=10, precision="16-mixed", val_check_interval=1/4)
    trainer.fit(classification_module, data_module)

def test():
    import pytorch_lightning.loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    csv_logger = pl_loggers.CSVLogger(save_dir="logs/")
    # trainer = Trainer(logger=[tb_logger, comet_logger])
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=3, monitor="val_auc", mode="max")
    
    classification_module = VideoClassificationLightningModule()
    data_module = DeepFakeDataModule()
    trainer = pytorch_lightning.Trainer(callbacks=[checkpoint_callback], logger=[tb_logger, csv_logger], max_epochs=10, precision="16-mixed", val_check_interval=1/4)
    # best 0.7009264881 1) version_80 step=1260, pretrain 2) load pretrain and train epoch=4-step=22688 3) version_108/checkpoints/epoch=4-step=22688.ckpt
    # trainer.test(classification_module, data_module, ckpt_path="logs/AV_ViTB/version_108/checkpoints/epoch=4-step=22688.ckpt")
    trainer.test(classification_module, data_module, ckpt_path="epoch=3-step=20166.ckpt")
    # trainer.predict()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    # train()
    test()