"""Newdata-only IGEVLS forward path with corrected rectification geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from core import igev_stereo as igev_base
from core.newdata_stereo_geometry import (
    CorrectedCombinedGeoEncodingVolume,
    build_corrected_sub_volume,
    corrected_disparity_regression,
    interleave_right_features,
)


class CorrectedNewDataIGEVLS(igev_base.IGEVLS):
    """Released architecture with corrected split-right inference geometry."""

    def _autocast(self, device: torch.device):
        enabled = bool(self.args.mixed_precision and device.type == "cuda")
        return torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, self.args.precision_dtype, torch.float16),
            enabled=enabled,
        )

    def forward(
        self,
        image1,
        image2_split,
        iters=12,
        flow_init=None,
        return_probability=False,
    ):
        del flow_init
        split_count = len(image2_split)
        if split_count != int(self.args.tau):
            raise ValueError(
                f"Received {split_count} right phases, but --tau={self.args.tau}"
            )
        if split_count % 2 == 0:
            raise ValueError(
                "Corrected newdata requires odd --tau; "
                "the calibrated dataset uses tau=3"
            )

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2_split = [
            (2 * (image2 / 255.0) - 1.0).contiguous() for image2 in image2_split
        ]

        with self._autocast(image1.device):
            features_left = self.feature(image1)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            features_left[0] = torch.cat((features_left[0], stem_4x), dim=1)
            match_left = self.desc(self.conv(features_left[0]))

            match_right_split = []
            for image2 in image2_split:
                features_right = self.feature(image2)
                stem_2y = self.stem_2(image2)
                stem_4y = self.stem_4(stem_2y)
                features_right[0] = torch.cat((features_right[0], stem_4y), dim=1)
                match_right_split.append(self.desc(self.conv(features_right[0])))

            cost_volume = build_corrected_sub_volume(
                match_left,
                match_right_split,
                self.args.max_disp // 4,
                num_groups=8,
            )
            cost_volume = self.corr_stem(cost_volume)
            cost_volume = self.corr_feature_att(cost_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(cost_volume, features_left)

            probability = F.softmax(
                self.classifier(geo_encoding_volume).squeeze(1),
                dim=1,
            )
            # Unit: one sample on the interleaved right stride-four feature grid.
            init_disp = corrected_disparity_regression(probability, keepdim=True)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(item[0]) for item in cnet_list]
            inp_list = [torch.relu(item[1]) for item in cnet_list]
            inp_list = [
                list(conv(item).split(split_size=conv.out_channels // 3, dim=1))
                for item, conv in zip(inp_list, self.context_zqr_convs)
            ]

        geo_fn = CorrectedCombinedGeoEncodingVolume(
            match_left.float(),
            interleave_right_features(match_right_split).float(),
            geo_encoding_volume.float(),
            split_count=split_count,
            radius=self.args.corr_radius,
            num_levels=self.args.corr_levels,
            dynamic_radius=self.args.dynamic_radius,
        )
        batch, _, height, width = match_left.shape
        left_coords = (
            torch.arange(width, dtype=torch.float32, device=match_left.device)
            .reshape(1, 1, width, 1)
            .repeat(batch, height, 1, 1)
        )
        disp = init_disp

        for iteration in range(iters):
            disp = disp.detach()
            geo_features = geo_fn(disp, left_coords)
            with self._autocast(image1.device):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list,
                    inp_list,
                    geo_features,
                    disp,
                    iter16=self.args.n_gru_layers == 3,
                    iter08=self.args.n_gru_layers >= 2,
                )
            disp = disp + delta_disp
            if iteration < iters - 1:
                continue

            # context upsampling converts stride-four units to input pixels;
            # divide by N because disp is measured on the N-times denser grid.
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x) / float(split_count)

        if return_probability:
            return disp_up, probability
        return disp_up
