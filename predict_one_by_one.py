import itertools
import logging
import math
from operator import index
import os
import random
import time
from turtle import forward
from typing import Any
import pandas as pd
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
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
import tqdm

import torch
import torch.nn.functional as F

import pytorchvideo.models.resnet
import torchmetrics

from models.av_vit import get_deepfake_av_model

T = 16

def make_deepfake_vitb():
    # pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(input_channels=3, spatial_size=256, )
    return get_deepfake_av_model()


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

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, tta=False, use_pretrained_model=False):
        super().__init__()
        self.av_model = make_deepfake_vitb()
        if use_pretrained_model:
            # https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth kinetics400 pretrained
            model_state = torch.load("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/MVIT_B_16x4.pyth")["model_state"]
            del model_state["head.proj.weight"]
            del model_state["head.proj.bias"]
            del model_state["cls_positional_encoding.pos_embed_spatial"]
            del model_state["cls_positional_encoding.pos_embed_temporal"]
            res = self.av_model.load_state_dict(model_state, strict=False)
            print(res)
        
        # def print_model_info_by_thop(model):
        #     try:
        #         from thop import clever_format
        #         from thop import profile
        #         input_video = torch.randn(1, 3, 16, 256, 256)
        #         input_audio = torch.randn(1, 1, 16, 128, 128)
        #         flops, params = profile(model, inputs=(input_video, input_audio), )
        #         flops, params = clever_format([flops, params], "%.3f")
        #         print("Model Info:", f"FLOPs is {flops}  Size of model is {params}")
        #     except:
        #         pass
            
        # print_model_info_by_thop(self.av_model)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=2)
        self.tta = tta
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
    
    def get_offset_video_frames(self, batch, frame_num=16):
        start = min(max(int(self.rng.gauss(2, 1) * 4), 0), 4)
        batch["video"] = batch["video"][:, :, start: start + frame_num]
        start = min(max(int(self.rng.gauss(2, 1) * 4), 0), 4)
        batch["audio"] = batch["audio"][:, :, start: start + frame_num]
        return batch
    
    def get_offset_video_frames_val(self, batch, frame_num=16):
        start = 0
        batch["video"] = batch["video"][:, :, start: start + frame_num]
        start = 0
        batch["audio"] = batch["audio"][:, :, start: start + frame_num]
        return batch

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        batch = self.get_offset_video_frames(batch, T)
        batch = self.shuffle_batch(batch, prob=0.2)
        batch_size = batch["video"].shape[0]
        y_hat = self.forward(batch["video"], batch["audio"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.get_offset_video_frames_val(batch, T)
        batch_size = batch["video"].shape[0]
        video_name = batch["video_name"]
        y_hat = self.forward(batch["video"], batch["audio"])
        loss = F.cross_entropy(y_hat, batch["label"])
        
        self.csv_res(y_hat, batch["label"], video_name)
        y_hat = F.softmax(y_hat, dim=-1)
        acc = self.val_accuracy(y_hat, batch["label"])
        auc = self.val_auc(y_hat, batch["label"])
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_auc", auc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def on_validation_start(self) -> None:
        self.csv_res.reset()
        return super().on_validation_start()
    
    def on_validation_end(self) -> None:
        self.csv_res.aggregate(os.path.join(self.logger.log_dir, f"merged_results_step_{self.current_epoch}-{self.global_step}.csv"))
        return super().on_validation_end()

    def _tta_transform(self, batch):
        tta_transform = RandomHorizontalFlip(p=1)
        
        return tta_transform(batch)

    def test_step(self, batch, batch_idx):
        video_name = batch["video_name"]
        
        with torch.no_grad():
            y_hat = self.forward(batch["video"], batch["audio"])
            if self.tta:
                y_hat_tta = self.forward(self._tta_transform(batch["video"]), batch["audio"])
                y_hat = (y_hat_tta + y_hat) / 2
        
        fake_prob = y_hat
        self.csv_res(fake_prob, batch["label"], video_name)
    
    def predict_step(self, batch) -> Any:
        y_hat = self.forward(batch["video"], batch["audio"])
        y_hat = F.softmax(y_hat, dim=-1)
        return y_hat
    
    def on_test_start(self) -> None:
        self.csv_res.reset()
        return super().on_test_start()
    
    def on_test_end(self) -> None:
        self.csv_res.aggregate(os.path.join(f"merged_results_testset_tta={self.tta}.csv"))
        return super().on_test_end()

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-5)

class VideoPredictor():
    
    def __init__(self, checkpoint, decode_audio: bool = True,
        decoder: str = "pyav", transforms = None, clip_duration=4, target_batch=4, fp_16=False, device="cuda:0") -> None:
        from pytorchvideo.data.video import VideoPathHandler
        
        classification_module = VideoClassificationLightningModule.load_from_checkpoint(checkpoint)
        self.model = classification_module
        classification_module.eval()

        self.video_path_handler = VideoPathHandler()
        self._decode_audio = decode_audio
        self._decoder = decoder
        self.clip_duration = clip_duration
        self._clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", clip_duration, None, True)
        self._transform = transforms
        if self._transform is None:
            self._transform = Compose(
                [
                    self._video_transform("val"),
                    self._audio_transform("val"),
                ]
            )
        self.device = device
        self.fp_16 = fp_16
        self._next_clip_start_time = 0
        self.model = self.model.to(self.device)
        if fp_16:
            self.model = self.model.half()
        
        # Cache
        self._loaded_video_label = None
        self.target_batch = target_batch
        # Logger
        self.logger = logging.getLogger("VideoPredictor")
        self.logger.setLevel(1)
    
    def _video_transform(self, mode="val"):
        """
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
        
        video_num_subsampled = T + 4
        if mode == "val":
            video_num_subsampled -= 4
        # video_means, video_stds = (0, 0, 0), (1, 1, 1)
        video_means, video_stds = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
        video_crop_size = 256
        video_horizontal_flip_p = 0.5
        video_min_short_side_scale, video_max_short_side_scale = 256, 320
        
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(video_num_subsampled),
                    # Lambda(lambda x: x / 255.0),
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

    def _audio_transform(self, mode="val"):
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
        audio_frame_num = T + 4 # frame_num + offset for augment
        if mode == "val":
            audio_frame_num -= 4
            
        audio_num_mels = 128
        audio_mel_num_subsample = audio_frame_num * audio_num_mels
        
        
        n_fft = int(
            float(audio_resampled_rate) / 1000 * audio_mel_window_size
        )
        hop_length = int(
            float(audio_resampled_rate) / 1000 * audio_mel_step_size
        )
        eps = 1e-10

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
                            ),  # (T, F) -> (1, T_num, T_time, F)
                            # ShortSideScale(
                            #     size=224,
                            # ),
                            
                            Normalize((audio_logmel_mean,), (audio_logmel_std,)),
                        ]
                    ),
                )
        
    def __call__(self, video_path) -> Any:
        res_list = []
        infer_cost = []
        MAX_FAILED = 20
        try:
            # self.logger.info(f"Start parse {video_path} ...")
            video = self.video_path_handler.video_from_path(
                            video_path,
                            decode_audio=self._decode_audio,
                            decoder=self._decoder,
                        )
            
            esm_duration = video.duration
            sample_batch_list = []
            max_iter = min(MAX_FAILED, math.ceil(esm_duration / self.clip_duration))
            bar = tqdm.trange(max_iter, leave=False)
            for i in bar:
                self._loaded_video_label = (video, 0)
                (
                    clip_start,
                    clip_end,
                    clip_index,
                    aug_index,
                    is_last_clip,
                ) = self._clip_sampler(
                    self._next_clip_start_time, video.duration, {}
                )
                
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)
                self._next_clip_start_time = clip_end
                video_is_null = (
                    self._loaded_clip is None or self._loaded_clip["video"] is None
                )
                if (
                        is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
                    ) or video_is_null:
                    # Close the loaded encoded video and reset the last sampled clip time ready
                    # to sample a new video on the next iteration.
                    self._loaded_video_label[0].close()
                    self._loaded_video_label = None
                    self._next_clip_start_time = 0.0
                    self._clip_sampler.reset()
                    if video_is_null:
                        self.logger.debug(
                            "Failed to load clip {}; trial {}".format(video.name, i)
                        )
                        
                        break
                    # self.logger.info(f"Clip {video.name}.")
                    bar.set_postfix_str(f"Clip {video.name}.")
                    # self.logger.info(f"Finished {video.name}")
                else:
                    bar.set_postfix_str(f"Clip {video.name}.")
                    
                frames = self._loaded_clip["video"]
                audio_samples = self._loaded_clip["audio"]
                sample_dict = {
                    "video": frames,
                    "video_name": video.name,
                    "video_index": 0,
                    "clip_index": clip_index,
                    "aug_index": aug_index,
                    **({"audio": audio_samples} if audio_samples is not None else {}),
                }
                if self._transform is not None:
                    sample_dict = self._transform(sample_dict)
                    
                if sample_dict is None:
                    self.logger.error(f"Transform failed, skip the video.")
                    return 0
                
                sample_dict["video"] = sample_dict["video"].to(self.device, non_blocking=True).unsqueeze(dim=0)
                sample_dict["audio"] = sample_dict["audio"].to(self.device, non_blocking=True).unsqueeze(dim=0)
                sample_batch_list.append(sample_dict)
                if len(sample_batch_list) >= self.target_batch:
                    bar.set_postfix_str("Inference batch...")
                    start = time.time_ns()
                    res = self._infer(sample_batch_list)
                    cost = (time.time_ns() - start) / 1e6
                    res_list+=res
                    infer_cost.append(cost)
                    sample_batch_list.clear()
            
            # Check if the list has extra data
            if len(sample_batch_list) > 0:
                start = time.time_ns()
                res = self._infer(sample_batch_list)
                cost = (time.time_ns() - start) / 1e6
                res_list+=res
                infer_cost.append(cost)
                sample_batch_list.clear()
                    
        except Exception as e:
            self.logger.error(f"{e}")
            return 0
        
        # self.logger.debug(f"Fake prob: {max(res_list):.04f}, Infer cost: {sum(infer_cost):.01f}ms")
        return max(res_list) if len(res_list) > 0 else 0

    def _batch_sample(self, sample_dict_list) -> dict:
        # print("batch_sample", len(sample_dict_list))
        video_batch_list = []
        audio_batch_list = []
        for i in sample_dict_list:
            video_batch = i["video"]
            audio_batch = i["audio"]
            video_batch_list.append(video_batch)
            audio_batch_list.append(audio_batch)
            
        output_dict = sample_dict_list[0]
        output_dict["video"] = torch.concat(video_batch_list)
        output_dict["audio"] = torch.concat(audio_batch_list)
        if self.fp_16:
            output_dict["video"] = output_dict["video"].half()
            output_dict["audio"] = output_dict["audio"].half()
        
        return output_dict

    def _infer(self, sample_dict_list) -> list:
        outputs = []
        with torch.no_grad():
            res = self.model.predict_step(self._batch_sample(sample_dict_list))
            res: torch.Tensor

            for one_ in res:
                fake_prob = one_[1].item()
                outputs.append(fake_prob)
        return outputs
    
