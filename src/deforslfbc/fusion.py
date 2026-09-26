"""Inference-only implementation of the released long/short-range fusion model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SCALE_CHANNELS = (32, 64, 128, 256)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: tuple[int, ...]):
        super().__init__()
        self.levels = len(out_channels)
        self.convs = nn.ModuleList()
        for out_channel in out_channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channel, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channel

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for convolution in self.convs:
            x = convolution(x)
            features.append(x)
        return features


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels_mono: int,
        feat_dim: int,
        hidden_dim: int,
        num_scales: int,
    ):
        super().__init__()
        channels = SCALE_CHANNELS[:num_scales]
        self.mono_conv = nn.Conv2d(in_channels_mono, feat_dim, kernel_size=1)
        self.depth_proj = nn.ModuleList(
            nn.Conv2d(channel, feat_dim, kernel_size=1) for channel in channels
        )
        self.rgb_proj = nn.ModuleList(
            nn.Conv2d(channel, feat_dim, kernel_size=1) for channel in channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                feat_dim * (1 + 2 * num_scales),
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

    def forward(
        self,
        mono_feature: torch.Tensor,
        depth_features: list[torch.Tensor],
        rgb_features: list[torch.Tensor],
    ) -> torch.Tensor:
        mono_projection = self.mono_conv(mono_feature)
        target_size = mono_projection.shape[-2:]
        depth_projections = []
        rgb_projections = []
        for index, feature in enumerate(depth_features):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=True
                )
            depth_projections.append(self.depth_proj[index](feature))
        for index, feature in enumerate(rgb_features):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=True
                )
            rgb_projections.append(self.rgb_proj[index](feature))
        return self.fusion(
            torch.cat(
                [mono_projection, *depth_projections, *rgb_projections], dim=1
            )
        )


class MultiHeadAttentionRegion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_feature: torch.Tensor,
        key_feature: torch.Tensor,
        value_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, dimension = query_feature.shape
        key_count = key_feature.shape[1]
        query = self.q_proj(query_feature).view(
            batch, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(key_feature).view(
            batch, key_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(value_feature).view(
            batch, key_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = self.dropout(F.softmax(attention, dim=-1))
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(
            batch, query_count, dimension
        )
        return self.out_proj(output)


class FusionNetV2(nn.Module):
    """Released model with corrected LUT axes, stereo FB, and explicit ROI."""

    def __init__(
        self,
        in_channels_mono: int = 768,
        in_channels_stereo: int = 72,
        hidden_dim: int = 128,
        num_regions: int = 64,
        num_lut_points: int = 96,
        max_depth: float = 100.0,
        num_heads: int = 4,
        feat_dim: int = 64,
        dropout: float = 0.1,
        num_scales: int = 4,
        border_width: int = 1,
        stereo_roi_x: int | None = None,
        stereo_roi_y: int | None = None,
    ):
        super().__init__()
        if not 1 <= num_scales <= len(SCALE_CHANNELS):
            raise ValueError("num_scales must be in [1, 4]")
        if border_width <= 0:
            raise ValueError("border_width must be positive")
        self.num_regions = num_regions
        self.num_lut_points = num_lut_points
        self.max_depth = max_depth
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.border_width = border_width
        self.stereo_roi_x = stereo_roi_x
        self.stereo_roi_y = stereo_roi_y
        self.center_start = num_regions // 4
        self.center_end = num_regions * 3 // 4
        self.center_size = self.center_end - self.center_start
        self.inner_center_start = self.center_start + border_width
        self.inner_center_end = self.center_end - border_width
        if self.inner_center_start >= self.inner_center_end:
            raise ValueError("border_width is too large")

        channels = SCALE_CHANNELS[:num_scales]
        self.depth_encoder = MultiScaleEncoder(1, channels)
        self.rgb_encoder = MultiScaleEncoder(3, channels)
        self.stereo_depth_encoder = MultiScaleEncoder(1, channels)
        self.region_encoder = FeatureEncoder(
            in_channels_mono, feat_dim, hidden_dim, num_scales
        )
        center_in_dim = in_channels_stereo + 3 * sum(channels)
        self.center_encoder = nn.Sequential(
            nn.Conv2d(center_in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.embed_dim = hidden_dim
        self.region_to_q = nn.Linear(hidden_dim, hidden_dim)
        self.center_to_k = nn.Linear(hidden_dim, hidden_dim)
        self.center_to_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention = MultiHeadAttentionRegion(hidden_dim, num_heads, dropout)
        self.lut_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_lut_points),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_regions * num_regions, hidden_dim) * 0.02
        )
        self.register_buffer("_weight_map_cache", None, persistent=False)
        self._weight_map_cache_key = None

    def _resolve_stereo_roi(
        self, height: int, width: int, stereo_height: int, stereo_width: int
    ) -> tuple[int, int]:
        start_h = (
            (height - stereo_height) // 2
            if self.stereo_roi_y is None
            else self.stereo_roi_y
        )
        start_w = (
            (width - stereo_width) // 2
            if self.stereo_roi_x is None
            else self.stereo_roi_x
        )
        if (
            start_h < 0
            or start_w < 0
            or start_h + stereo_height > height
            or start_w + stereo_width > width
        ):
            raise ValueError(
                "Stereo ROI does not fit the full image: "
                f"origin=({start_w},{start_h}), stereo={stereo_width}x{stereo_height}, "
                f"full={width}x{height}"
            )
        return start_h, start_w

    def _region_bounds(
        self, height: int, width: int, stereo_height: int, stereo_width: int
    ) -> tuple[int, int, int, int]:
        start_h, start_w = self._resolve_stereo_roi(
            height, width, stereo_height, stereo_width
        )
        coordinates = (
            start_h * self.num_regions / height,
            (start_h + stereo_height) * self.num_regions / height,
            start_w * self.num_regions / width,
            (start_w + stereo_width) * self.num_regions / width,
        )
        rounded = tuple(int(round(value)) for value in coordinates)
        if any(
            abs(value - integer) > 1e-6
            for value, integer in zip(coordinates, rounded)
        ):
            raise ValueError("Stereo ROI boundaries must align with the region grid")
        row_start, row_end, column_start, column_end = rounded
        if (
            row_end - row_start != self.center_size
            or column_end - column_start != self.center_size
        ):
            raise ValueError("Stereo ROI must cover the configured center grid size")
        return row_start, row_end, column_start, column_end

    def _compute_weight_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        stereo_height: int,
        stereo_width: int,
    ) -> torch.Tensor:
        start_h, start_w = self._resolve_stereo_roi(
            height, width, stereo_height, stereo_width
        )
        end_h = start_h + stereo_height
        end_w = start_w + stereo_width
        ratio_h = stereo_height / self.center_size
        ratio_w = stereo_width / self.center_size
        inner_start_h = start_h + (
            self.inner_center_start - self.center_start
        ) * ratio_h
        inner_end_h = start_h + (
            self.inner_center_end - self.center_start
        ) * ratio_h
        inner_start_w = start_w + (
            self.inner_center_start - self.center_start
        ) * ratio_w
        inner_end_w = start_w + (
            self.inner_center_end - self.center_start
        ) * ratio_w
        border_width_h = self.border_width * ratio_w
        border_width_v = self.border_width * ratio_h
        y = torch.arange(height, device=device).float()
        x = torch.arange(width, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        normalized = torch.stack(
            [
                torch.clamp(inner_start_w - xx, min=0) / border_width_h,
                torch.clamp(xx - inner_end_w, min=0) / border_width_h,
                torch.clamp(inner_start_h - yy, min=0) / border_width_v,
                torch.clamp(yy - inner_end_h, min=0) / border_width_v,
            ]
        )
        weight = torch.clamp(torch.max(normalized, dim=0)[0], 0, 1)
        stereo_mask = (
            (yy >= start_h) & (yy < end_h) & (xx >= start_w) & (xx < end_w)
        )
        weight = torch.where(stereo_mask, weight, torch.ones_like(weight))
        return weight.unsqueeze(0).unsqueeze(0)

    def _build_lut_grid(self, mono_depth: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = mono_depth.shape
        depth_coordinate = torch.clamp(mono_depth / self.max_depth, 0, 1) * 2 - 1
        y_coordinate = torch.linspace(
            -1, 1, height, device=mono_depth.device, dtype=mono_depth.dtype
        ).view(1, height, 1).expand(batch, height, width)
        x_coordinate = torch.linspace(
            -1, 1, width, device=mono_depth.device, dtype=mono_depth.dtype
        ).view(1, 1, width).expand(batch, height, width)
        # For a [B,C,D,H,W] input, grid_sample expects (x=W,y=H,z=D).
        return torch.stack(
            [x_coordinate, y_coordinate, depth_coordinate.squeeze(1)], dim=-1
        ).unsqueeze(1)

    def _get_weight_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        stereo_height: int,
        stereo_width: int,
    ) -> torch.Tensor:
        cache_key = (
            height,
            width,
            stereo_height,
            stereo_width,
            device.type,
            device.index,
        )
        if self._weight_map_cache is None or self._weight_map_cache_key != cache_key:
            self._weight_map_cache = self._compute_weight_map(
                height, width, device, stereo_height, stereo_width
            )
            self._weight_map_cache_key = cache_key
        return self._weight_map_cache

    def forward(
        self,
        mono_depth: torch.Tensor,
        mono_feature: torch.Tensor,
        stereo_depth: torch.Tensor,
        stereo_feature: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        batch = mono_depth.shape[0]
        height, width = mono_depth.shape[-2:]
        stereo_feature_height, stereo_feature_width = stereo_feature.shape[-2:]
        stereo_height, stereo_width = stereo_depth.shape[-2:]
        roi_start_h, roi_start_w = self._resolve_stereo_roi(
            height, width, stereo_height, stereo_width
        )

        depth_multi = self.depth_encoder(mono_depth)
        rgb_multi = self.rgb_encoder(rgb)
        stereo_depth_multi = self.stereo_depth_encoder(stereo_depth)
        region_feature = self.region_encoder(mono_feature, depth_multi, rgb_multi)
        region_grid = F.adaptive_avg_pool2d(
            region_feature, (self.num_regions, self.num_regions)
        )
        region_grid = region_grid.permute(0, 2, 3, 1).reshape(
            batch, -1, self.embed_dim
        ) + self.pos_embedding

        row_start, row_end, column_start, column_end = self._region_bounds(
            height, width, stereo_height, stereo_width
        )
        center_indices = torch.tensor(
            [
                row * self.num_regions + column
                for row in range(row_start, row_end)
                for column in range(column_start, column_end)
            ],
            device=region_grid.device,
        )
        center_feature = region_grid[:, center_indices, :]
        key = self.center_to_k(center_feature)
        query = self.region_to_q(region_grid)

        target_size = (stereo_feature_height * 2, stereo_feature_width * 2)
        rgb_center_list = []
        depth_center_list = []
        residual_center_list = []
        for scale in range(self.num_scales):
            mono_height, mono_width = depth_multi[scale].shape[-2:]
            stereo_scale_height, stereo_scale_width = stereo_depth_multi[
                scale
            ].shape[-2:]
            start_h = int(round(roi_start_h * mono_height / height))
            start_w = int(round(roi_start_w * mono_width / width))
            if (
                start_h + stereo_scale_height > mono_height
                or start_w + stereo_scale_width > mono_width
            ):
                raise ValueError("Stereo ROI does not fit a feature pyramid level")
            mono_center = depth_multi[scale][
                :,
                :,
                start_h : start_h + stereo_scale_height,
                start_w : start_w + stereo_scale_width,
            ]
            rgb_center = rgb_multi[scale][
                :,
                :,
                start_h : start_h + stereo_scale_height,
                start_w : start_w + stereo_scale_width,
            ]
            residual = mono_center - stereo_depth_multi[scale]
            rgb_center_list.append(
                F.interpolate(
                    rgb_center, size=target_size, mode="bilinear", align_corners=True
                )
            )
            depth_center_list.append(
                F.interpolate(
                    stereo_depth_multi[scale],
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                )
            )
            residual_center_list.append(
                F.interpolate(
                    residual, size=target_size, mode="bilinear", align_corners=True
                )
            )

        center_input = torch.cat(
            [
                F.interpolate(
                    stereo_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                ),
                torch.cat(rgb_center_list, dim=1),
                torch.cat(depth_center_list, dim=1),
                torch.cat(residual_center_list, dim=1),
            ],
            dim=1,
        )
        center_feature_map = self.center_encoder(center_input)
        center_grid_feature = F.adaptive_avg_pool2d(
            center_feature_map, (self.center_size, self.center_size)
        )
        center_value = center_grid_feature.permute(0, 2, 3, 1).reshape(
            batch, -1, self.embed_dim
        )
        attention_output = self.attention(
            query, key, self.center_to_v(center_value)
        )

        lut = self.lut_mlp(attention_output).view(
            batch, self.num_regions, self.num_regions, self.num_lut_points
        )
        lut = lut.permute(0, 3, 1, 2).unsqueeze(1)
        residual_depth = F.grid_sample(
            lut, self._build_lut_grid(mono_depth), mode="bilinear", align_corners=True
        ).squeeze(2)
        valid_region_mask = mono_depth < (1.5 * self.max_depth)
        residual_depth = residual_depth * valid_region_mask.float()
        corrected_depth = residual_depth + mono_depth
        corrected_depth[valid_region_mask] = torch.clamp(
            corrected_depth[valid_region_mask], 0, self.max_depth
        )

        stereo_depth_placed = torch.zeros_like(corrected_depth)
        stereo_depth_placed[
            :,
            :,
            roi_start_h : roi_start_h + stereo_height,
            roi_start_w : roi_start_w + stereo_width,
        ] = stereo_depth
        final_weight = self._get_weight_map(
            height, width, mono_depth.device, stereo_height, stereo_width
        ).expand(batch, -1, -1, -1)
        final_depth = (
            final_weight * corrected_depth
            + (1 - final_weight) * stereo_depth_placed
        )
        final_depth[valid_region_mask] = torch.clamp(
            final_depth[valid_region_mask], 0, self.max_depth
        )
        return final_depth
