"""Inference architecture shared by the released corrected stereo model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.extractor import Feature, MultiBasicEncoder
from core.submodule import (
    BasicConv,
    BasicConv_IN,
    Conv2x,
    Conv2x_IN,
    FeatureAtt,
    context_upsample,
)
from core.update import BasicMultiUpdateBlock


class Hourglass(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )
        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )
        self.conv3 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 6,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )
        self.conv3_up = BasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv1_up = BasicConv(
            in_channels * 2,
            8,
            deconv=True,
            is_3d=True,
            bn=False,
            relu=False,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.agg_0 = nn.Sequential(
            BasicConv(
                in_channels * 8,
                in_channels * 4,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.agg_1 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 2,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.feature_att_8 = FeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, 64)

    def forward(self, tensor, features):
        conv1 = self.feature_att_8(self.conv1(tensor), features[1])
        conv2 = self.feature_att_16(self.conv2(conv1), features[2])
        conv3 = self.feature_att_32(self.conv3(conv2), features[3])

        conv2 = torch.cat((self.conv3_up(conv3), conv2), dim=1)
        conv2 = self.feature_att_up_16(self.agg_0(conv2), features[2])
        conv1 = torch.cat((self.conv2_up(conv2), conv1), dim=1)
        conv1 = self.feature_att_up_8(self.agg_1(conv1), features[1])
        return self.conv1_up(conv1)


class IGEVLS(nn.Module):
    """Layers required by the split-right inference implementation."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn="batch",
            downsample=args.n_downsample,
        )
        self.update_block = BasicMultiUpdateBlock(
            self.args,
            hidden_dims=args.hidden_dims,
        )
        self.context_zqr_convs = nn.ModuleList(
            nn.Conv2d(context_dims[index], args.hidden_dims[index] * 3, 3, padding=1)
            for index in range(self.args.n_gru_layers)
        )
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )

        # Retained for strict compatibility with the released state dictionary.
        # Only the GRU upsampling path is used during inference.
        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1)
        )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1)
        )

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.corr_stem = BasicConv(
            8,
            8,
            is_3d=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = Hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        with torch.autocast(
            device_type=disp.device.type,
            enabled=bool(
                self.args.mixed_precision and disp.device.type == "cuda"
            ),
            dtype=getattr(torch, self.args.precision_dtype, torch.float16),
        ):
            feature = self.spx_2_gru(mask_feat_4, stem_2x)
            weights = F.softmax(self.spx_gru(feature), dim=1)
            return context_upsample(disp * 4.0, weights).unsqueeze(1)
