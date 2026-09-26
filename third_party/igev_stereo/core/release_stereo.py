"""Inference-only construction of the released corrected IGEV-Stereo model."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from core import igev_stereo as igev_base
from core.submodule import BasicConv_IN, Conv2x_IN


class ReleaseMobileNetV2Feature(nn.Module):
    """Feature pyramid whose complete weights come from ``stereo.pth``."""

    def __init__(self):
        super().__init__()
        model = timm.create_model(
            "mobilenetv2_100",
            pretrained=False,
            features_only=True,
        )
        layers = [1, 2, 3, 5, 6]
        channels = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = nn.ReLU6(inplace=True)
        self.block0 = nn.Sequential(*model.blocks[0 : layers[0]])
        self.block1 = nn.Sequential(*model.blocks[layers[0] : layers[1]])
        self.block2 = nn.Sequential(*model.blocks[layers[1] : layers[2]])
        self.block3 = nn.Sequential(*model.blocks[layers[2] : layers[3]])
        self.block4 = nn.Sequential(*model.blocks[layers[3] : layers[4]])

        self.deconv32_16 = Conv2x_IN(
            channels[4], channels[3], deconv=True, concat=True
        )
        self.deconv16_8 = Conv2x_IN(
            channels[3] * 2, channels[2], deconv=True, concat=True
        )
        self.deconv8_4 = Conv2x_IN(
            channels[2] * 2, channels[1], deconv=True, concat=True
        )
        self.conv4 = BasicConv_IN(
            channels[1] * 2,
            channels[1] * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, image: torch.Tensor):
        image = self.act1(self.bn1(self.conv_stem(image)))
        feature_2 = self.block0(image)
        feature_4 = self.block1(feature_2)
        feature_8 = self.block2(feature_4)
        feature_16 = self.block3(feature_8)
        feature_32 = self.block4(feature_16)

        feature_16 = self.deconv32_16(feature_32, feature_16)
        feature_8 = self.deconv16_8(feature_16, feature_8)
        feature_4 = self.deconv8_4(feature_8, feature_4)
        feature_4 = self.conv4(feature_4)
        return [feature_4, feature_8, feature_16, feature_32]


def build_release_stereo(args):
    """Construct the release architecture without training/data adapters."""
    from core.newdata_igevls import CorrectedNewDataIGEVLS

    original_feature = igev_base.Feature
    igev_base.Feature = ReleaseMobileNetV2Feature
    try:
        return CorrectedNewDataIGEVLS(args)
    finally:
        igev_base.Feature = original_feature
