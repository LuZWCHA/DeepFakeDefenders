
from models.dinov2_models import LinearClassifierWrapper, LinearClassifierWrapper2
from .resnet import *
from .resnet_ori import resnet50 as resnet50_ori
from .resnet_ori import resnet101 as resnet101_ori
from .resnet_ori_npr import resnet50 as resnet50_ori_npr
from .efficientnet import EfficientNet, DualSteamEfficientNet
from .efficientnet_npr import EfficientNet as EfficientNet_NPR
from .efficientnet_npr_grad import EfficientNet as EfficientNet_NPR_GRAD

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)

import torch

# # DINOv2
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# # DINOv2 with registers
# dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
# dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
# dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
# dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')


def get_model(model_name, classes_num=2):
    if model_name == "resnet18":
        return resnet18(pretrained=False)
    elif model_name == "resnet34":
        return resnet34(pretrained=False)
    elif model_name == "resnet50":
        return resnet50(pretrained=False)
    elif model_name == "resnet101":
        return resnet101(pretrained=False)
    elif model_name == "resnet101_ori":
        return resnet101_ori(pretrained=False)
    elif model_name == "resnet50_ori":
        return resnet50_ori(pretrained=False)
    elif model_name == "resnet50_ori_npr":
        return resnet50_ori_npr(pretrained=False)
    elif model_name == "efficientnet_b2":
        return EfficientNet.from_name("efficientnet-b2", num_classes=2)
    elif model_name == "efficientnet_b0":
        return EfficientNet.from_name("efficientnet-b0", num_classes=2)
    elif model_name == "efficientnet_b2_npr":
        return EfficientNet_NPR.from_name("efficientnet-b2", num_classes=2)
    elif model_name == "efficientnet_b2_npr_grad":
        return EfficientNet_NPR_GRAD.from_name("efficientnet-b2", num_classes=2)
    elif model_name == "efficientnet_b0_npr":
        return EfficientNet_NPR.from_name("efficientnet-b0", num_classes=2)
    elif model_name == "efficientnet_b0_ds":
        return DualSteamEfficientNet.from_name("efficientnet-b0", num_classes=2)
    elif model_name == "efficientnet_b2_ds":
        return DualSteamEfficientNet.from_name("efficientnet-b2", num_classes=2)
    elif model_name == "efficientnet_b0_npr_grad":
        return EfficientNet_NPR_GRAD.from_name("efficientnet-b0", num_classes=2)
    elif model_name == "efficientnet_b4_npr":
        return EfficientNet_NPR.from_name("efficientnet-b4", num_classes=2)
    elif model_name == "dinov2_vitb14":
        return LinearClassifierWrapper("dinov2_vitb14", n_classes=2)
    elif model_name == "dinov2_vitb14_nf":
        return LinearClassifierWrapper2("dinov2_vitb14", n_classes=2)