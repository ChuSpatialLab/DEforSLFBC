"""Rectification-aware geometry for split-right stereo inference.

For ``N`` horizontally split right views (``N`` is ``Nu`` in the derivation),
the interleaved right feature coordinate corresponding to left feature column
``x`` and non-negative cost index ``q`` is

    right_coordinate = N * x + (N - 1) / 2 - q.

The current corrected dataset has ``N == 3``.  Thus cost index zero uses phase
1, and increasing cost indices use phases ``1, 0, 2, 1, 0, 2, ...``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from core.submodule import groupwise_correlation
from core.utils.utils import bilinear_sampler


RIGHT_FEATURE_SPLITS = 3
LEFT_FEATURE_STRIDE = 4


def _validate_odd_split_count(split_count: int) -> None:
    if split_count <= 0 or split_count % 2 == 0:
        raise ValueError(
            "Corrected newdata currently requires an odd horizontal split count; "
            f"got N={split_count}. Even N needs half-index interpolation."
        )


def cost_index_to_phase_shift(cost_index: int, split_count: int) -> tuple[int, int]:
    """Return ``(phase, local_shift)`` for one monotonically increasing cost index.

    ``local_shift`` means that left column ``x`` is compared to the selected
    phase at column ``x - local_shift``.  The returned values exactly satisfy

    ``N * (x - local_shift) + phase == N*x + center - cost_index``.
    """

    _validate_odd_split_count(split_count)
    if cost_index < 0:
        raise ValueError(f"cost_index must be non-negative, got {cost_index}")
    center_phase = split_count // 2
    phase = (center_phase - cost_index) % split_count
    local_shift = (cost_index - center_phase + phase) // split_count
    return phase, local_shift


def cost_index_to_full_resolution_disparity(
    cost_index: torch.Tensor | float | int,
    split_count: int,
    feature_stride: int = LEFT_FEATURE_STRIDE,
):
    """Convert a cost/GRU disparity index to full-resolution left-image pixels."""

    _validate_odd_split_count(split_count)
    return cost_index * (float(feature_stride) / float(split_count))


def build_corrected_sub_volume(
    refimg_fea: torch.Tensor,
    targetimg_feas: list[torch.Tensor] | tuple[torch.Tensor, ...],
    maxdisp_left_feature: int,
    num_groups: int,
) -> torch.Tensor:
    """Build a cost volume whose depth index is physical right-grid disparity.

    A depth index ``q`` represents ``q`` samples on the interleaved right
    feature grid, or ``4*q/N`` pixels in the full-resolution left image for the
    stride-four matching features used by IGEVLS.
    """

    batch, channels, height, width = refimg_fea.shape
    split_count = len(targetimg_feas)
    _validate_odd_split_count(split_count)
    if maxdisp_left_feature <= 0:
        raise ValueError(f"maxdisp_left_feature must be positive, got {maxdisp_left_feature}")
    if channels % num_groups != 0:
        raise ValueError(f"channels={channels} is not divisible by groups={num_groups}")
    expected_shape = refimg_fea.shape
    for phase, target in enumerate(targetimg_feas):
        if target.shape != expected_shape:
            raise ValueError(
                f"right phase {phase} shape {tuple(target.shape)} != left shape {tuple(expected_shape)}"
            )

    depth = maxdisp_left_feature * split_count
    volume = refimg_fea.new_zeros([batch, num_groups, depth, height, width])
    for cost_index in range(depth):
        phase, local_shift = cost_index_to_phase_shift(cost_index, split_count)
        if local_shift >= width:
            continue
        if local_shift == 0:
            volume[:, :, cost_index] = groupwise_correlation(
                refimg_fea,
                targetimg_feas[phase],
                num_groups,
            )
        else:
            volume[:, :, cost_index, :, local_shift:] = groupwise_correlation(
                refimg_fea[:, :, :, local_shift:],
                targetimg_feas[phase][:, :, :, : width - local_shift],
                num_groups,
            )
    return volume.contiguous()


def corrected_disparity_regression(
    probability: torch.Tensor,
    keepdim: bool = True,
) -> torch.Tensor:
    """Soft-argmax in interleaved right-feature-grid index units."""

    if probability.ndim != 4:
        raise ValueError(f"Expected [B,D,H,W] probability, got {tuple(probability.shape)}")
    depth = probability.shape[1]
    cost_indices = torch.arange(
        depth,
        dtype=probability.dtype,
        device=probability.device,
    )
    cost_indices = cost_indices.view(1, depth, 1, 1)
    return torch.sum(probability * cost_indices, dim=1, keepdim=keepdim)


def interleave_right_features(
    match_right_split: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Reconstruct the high-rate horizontal right-feature sequence.

    Output order is ``phase0(x0), phase1(x0), ..., phaseN-1(x0), phase0(x1), ...``.
    """

    if not match_right_split:
        raise ValueError("match_right_split must not be empty")
    reference_shape = match_right_split[0].shape
    if any(item.shape != reference_shape for item in match_right_split):
        raise ValueError("All right phase feature tensors must have identical shapes")
    stacked = torch.stack(match_right_split, dim=-1)
    batch, channels, height, width, split_count = stacked.shape
    return stacked.reshape(batch, channels, height, width * split_count).contiguous()


class CorrectedCombinedGeoEncodingVolume:
    """IGEV recurrent lookup using corrected split-right coordinate geometry."""

    def __init__(
        self,
        init_fmap1: torch.Tensor,
        init_fmap2_interleaved: torch.Tensor,
        geo_volume: torch.Tensor,
        split_count: int,
        num_levels: int = 2,
        radius: int = 4,
        dynamic_radius: bool = False,
    ):
        _validate_odd_split_count(split_count)
        self.split_count = split_count
        self.center_phase = split_count // 2
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        init_corr = self.corr(init_fmap1, init_fmap2_interleaved)
        batch, height, width, _, right_width = init_corr.shape
        geo_batch, channels, _, geo_height, geo_width = geo_volume.shape
        if (geo_batch, geo_height, geo_width) != (batch, height, width):
            raise ValueError("Geometry volume and feature-map spatial shapes do not agree")

        geo_level = geo_volume.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, channels, 1, geo_volume.shape[2]
        )
        corr_level = init_corr.reshape(batch * height * width, 1, 1, right_width)
        self.geo_volume_pyramid.append(geo_level)
        self.init_corr_pyramid.append(corr_level)
        for _ in range(num_levels - 1):
            geo_level = F.avg_pool2d(geo_level, [1, 2], stride=[1, 2])
            corr_level = F.avg_pool2d(corr_level, [1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_level)
            self.init_corr_pyramid.append(corr_level)

        if dynamic_radius:
            self.radius = [radius // (2**level) for level in range(num_levels)]
        else:
            self.radius = [radius] * num_levels

    def __call__(self, disp: torch.Tensor, left_coords: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = disp.shape
        flattened_disp = disp.reshape(batch * height * width, 1, 1, 1)
        flattened_left = left_coords.reshape(batch * height * width, 1, 1, 1)
        outputs = []

        for level in range(self.num_levels):
            radius = self.radius[level]
            dx = torch.linspace(
                -radius,
                radius,
                2 * radius + 1,
                dtype=disp.dtype,
                device=disp.device,
            ).view(1, 1, 2 * radius + 1, 1)
            geo_x = flattened_disp / (2**level) + dx
            geo_coords = torch.cat([geo_x, torch.zeros_like(geo_x)], dim=-1)
            geo_sample = bilinear_sampler(self.geo_volume_pyramid[level], geo_coords)
            outputs.append(geo_sample.view(batch, height, width, -1))

            # q=0 is the center phase at N*x+center. Positive q moves left.
            right_zero = self.split_count * flattened_left + self.center_phase
            corr_x = (right_zero - flattened_disp) / (2**level) + dx
            corr_coords = torch.cat([corr_x, torch.zeros_like(corr_x)], dim=-1)
            corr_sample = bilinear_sampler(self.init_corr_pyramid[level], corr_coords)
            outputs.append(corr_sample.view(batch, height, width, -1))

        return torch.cat(outputs, dim=-1).permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, channels, height, left_width = fmap1.shape
        if fmap2.shape[:3] != (batch, channels, height):
            raise ValueError("Left and right correlation feature shapes are incompatible")
        right_width = fmap2.shape[-1]
        corr = torch.einsum("bchw,bchv->bhwv", fmap1, fmap2)
        return corr.reshape(batch, height, left_width, 1, right_width).contiguous()
