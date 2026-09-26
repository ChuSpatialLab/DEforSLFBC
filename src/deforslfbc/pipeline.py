"""End-to-end left/right RGB to full-frame metric depth inference."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .fusion import FusionNetV2


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = REPOSITORY_ROOT / "third_party"
IGEV_ROOT = THIRD_PARTY_ROOT / "igev_stereo"
for dependency_root in (THIRD_PARTY_ROOT, IGEV_ROOT):
    dependency = str(dependency_root)
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

MONO_FORMAT = "deforslfbc-mono-inference-v1"
STEREO_FORMAT = "deforslfbc-stereo-inference-v1"
FUSION_FORMAT = "deforslfbc-fusion-inference-v1"
EXPECTED_LEFT_SIZE = (768, 1280)
EXPECTED_RIGHT_SIZE = (1152, 1920)
EXPECTED_STEREO_SIZE = (384, 640)
RIGHT_PHASE_COUNT = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_module_prefix(state_dict: dict[str, torch.Tensor]):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def read_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return np.array(image, dtype=np.uint8, copy=True, order="C")


def rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)


@dataclass(frozen=True)
class SamplePaths:
    left: Path
    right: Path
    ground_truth: Path | None = None
    sky_mask: Path | None = None
    sample_id: str | None = None

    @classmethod
    def from_directory(cls, directory: Path) -> "SamplePaths":
        directory = Path(directory).expanduser().resolve()
        ground_truth = directory / "ground_truth.png"
        sky_mask = directory / "sky_mask.png"
        return cls(
            left=directory / "left.png",
            right=directory / "right.png",
            ground_truth=ground_truth if ground_truth.is_file() else None,
            sky_mask=sky_mask if sky_mask.is_file() else None,
            sample_id=directory.name,
        )

    def validate(self) -> None:
        paths = (self.left, self.right)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing input images: {missing}")


def preprocess_right_image(
    right_rgb: np.ndarray,
    phase_count: int = RIGHT_PHASE_COUNT,
) -> list[np.ndarray]:
    """Match the released dataset generator's vertical resize and phase split."""
    if right_rgb.shape != (*EXPECTED_RIGHT_SIZE, 3):
        raise ValueError(
            f"Expected right image {EXPECTED_RIGHT_SIZE[::-1]} RGB, "
            f"got shape {right_rgb.shape}"
        )
    if phase_count != RIGHT_PHASE_COUNT:
        raise ValueError(
            f"The released stereo model requires {RIGHT_PHASE_COUNT} right phases, "
            f"got {phase_count}"
        )
    stereo_height, stereo_width = EXPECTED_STEREO_SIZE
    vertical = cv2.resize(
        right_rgb,
        (right_rgb.shape[1], stereo_height),
        interpolation=cv2.INTER_LINEAR,
    )
    phases = [
        np.ascontiguousarray(vertical[:, phase::phase_count])
        for phase in range(phase_count)
    ]
    for phase, image in enumerate(phases):
        if image.shape != (stereo_height, stereo_width, 3):
            raise RuntimeError(
                f"Generated right phase {phase} has shape {image.shape}; "
                f"expected {(stereo_height, stereo_width, 3)}"
            )
    return phases


class DepthFusionPipeline:
    def __init__(
        self,
        checkpoint_root: Path = REPOSITORY_ROOT / "checkpoints",
        device: str = "cuda:0",
    ):
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            torch.cuda.set_device(self.device)
        self.mono_checkpoint = self.checkpoint_root / "mono.pth"
        self.stereo_checkpoint = self.checkpoint_root / "stereo.pth"
        self.fusion_checkpoint = self.checkpoint_root / "fusion.pth"
        for path in (
            self.mono_checkpoint,
            self.stereo_checkpoint,
            self.fusion_checkpoint,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing release checkpoint: {path}")

        self.mono_model, self.mono_config = self._load_mono()
        (
            self.stereo_model,
            self.stereo_args,
            self.stereo_calibration,
        ) = self._load_stereo()
        self.fusion_model, self.fusion_metadata = self._load_fusion()
        self.stereo_fb = float(self.stereo_calibration["fb_px_m"])
        data_config = self.fusion_metadata["data_config"]
        if abs(self.stereo_fb - float(data_config["stereo_fb"])) > 1e-9:
            raise RuntimeError("Stereo and fusion checkpoints disagree on fb")
        model_config = self.fusion_metadata["model_config"]
        self.stereo_roi_x = int(model_config["stereo_roi_x"])
        self.stereo_roi_y = int(model_config["stereo_roi_y"])
        self.max_depth = float(model_config["max_depth"])

    def _load_mono(self):
        from depth_anything_v2.dpt import DepthAnythingV2

        checkpoint = torch.load(
            self.mono_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if checkpoint.get("format") != MONO_FORMAT:
            raise RuntimeError("Unsupported mono checkpoint format")
        config = dict(checkpoint["model_config"])
        model = DepthAnythingV2(**config)
        model.load_state_dict(
            strip_module_prefix(checkpoint["model_state_dict"]), strict=True
        )
        return model.to(self.device).eval(), config

    def _load_stereo(self):
        from core.release_stereo import build_release_stereo

        checkpoint = torch.load(
            self.stereo_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if checkpoint.get("format") != STEREO_FORMAT:
            raise RuntimeError("Unsupported stereo checkpoint format")
        model_args = dict(checkpoint["model_args"])
        args = SimpleNamespace(**model_args)
        model = build_release_stereo(args)
        model.load_state_dict(
            strip_module_prefix(checkpoint["model_state_dict"]), strict=True
        )
        return model.to(self.device).eval(), args, dict(checkpoint["calibration"])

    def _load_fusion(self):
        checkpoint = torch.load(
            self.fusion_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if checkpoint.get("format") != FUSION_FORMAT:
            raise RuntimeError("Unsupported fusion checkpoint format")
        model = FusionNetV2(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.to(self.device).eval(), checkpoint

    @torch.inference_mode()
    def infer_mono(self, left_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        raw_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        height, width = raw_bgr.shape[:2]
        image, original_size = self.mono_model.image2tensor(raw_bgr, input_size=518)
        image = image.to(self.device)
        patch_height, patch_width = image.shape[-2] // 14, image.shape[-1] // 14
        intermediate = self.mono_model.pretrained.get_intermediate_layers(
            image,
            self.mono_model.intermediate_layer_idx[self.mono_model.encoder],
            return_class_token=True,
        )
        depth = self.mono_model.depth_head(
            intermediate, patch_height, patch_width
        ) * self.mono_model.max_depth
        depth = F.interpolate(
            depth, original_size, mode="bilinear", align_corners=True
        ).float()
        patch_tokens = intermediate[0][0]
        feature = patch_tokens.permute(0, 2, 1).reshape(
            1, patch_tokens.shape[-1], patch_height, patch_width
        )
        feature = F.interpolate(
            feature,
            (height // 4, width // 4),
            mode="bilinear",
            align_corners=True,
        ).float()
        return depth, feature

    @torch.inference_mode()
    def infer_stereo(
        self, left_crop: np.ndarray, right_phases: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from core.utils.utils import InputPadder

        left = rgb_tensor(left_crop, self.device)
        right = [rgb_tensor(image, self.device) for image in right_phases]
        padder = InputPadder(left.shape, divis_by=32)
        padded = padder.pad(left, *right)
        disparity, probability = self.stereo_model(
            image1=padded[0],
            image2_split=padded[1:],
            iters=int(self.stereo_args.valid_iters),
            return_probability=True,
        )
        disparity = padder.unpad(disparity, down=1).float()
        probability = padder.unpad(probability, down=4).float()
        depth = self.stereo_fb / torch.clamp(disparity, min=0.1)
        return disparity, probability, depth

    @torch.inference_mode()
    def predict(self, sample: SamplePaths) -> dict[str, torch.Tensor]:
        sample.validate()
        left_rgb = read_rgb(sample.left)
        if left_rgb.shape[:2] != EXPECTED_LEFT_SIZE:
            raise ValueError(
                f"Expected left image {EXPECTED_LEFT_SIZE[::-1]}, "
                f"got {left_rgb.shape[1]}x{left_rgb.shape[0]}"
            )
        right_rgb = read_rgb(sample.right)
        right_phases = preprocess_right_image(
            right_rgb, phase_count=int(self.stereo_args.tau)
        )
        y0, x0 = self.stereo_roi_y, self.stereo_roi_x
        stereo_height, stereo_width = EXPECTED_STEREO_SIZE
        left_crop = left_rgb[y0 : y0 + stereo_height, x0 : x0 + stereo_width]
        if left_crop.shape[:2] != EXPECTED_STEREO_SIZE:
            raise RuntimeError("Configured stereo ROI does not fit the left image")

        mono_depth, mono_feature = self.infer_mono(left_rgb)
        disparity, stereo_feature, stereo_depth = self.infer_stereo(
            left_crop, right_phases
        )
        left_tensor = rgb_tensor(left_rgb, self.device) / 255.0
        fused_depth = self.fusion_model(
            mono_depth,
            mono_feature,
            stereo_depth,
            stereo_feature,
            left_tensor,
        )
        result = {
            "depth": fused_depth[0, 0].float().cpu(),
            "mono_depth": mono_depth[0, 0].float().cpu(),
            "stereo_depth": stereo_depth[0, 0].float().cpu(),
            "stereo_disparity": disparity[0, 0].float().cpu(),
        }
        for name, tensor in result.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{name} contains non-finite values")
        return result

    def checkpoint_provenance(self) -> dict:
        return {
            "mono": {
                "file": self.mono_checkpoint.name,
                "sha256": sha256_file(self.mono_checkpoint),
            },
            "stereo": {
                "file": self.stereo_checkpoint.name,
                "sha256": sha256_file(self.stereo_checkpoint),
            },
            "fusion": {
                "file": self.fusion_checkpoint.name,
                "sha256": sha256_file(self.fusion_checkpoint),
            },
            "stereo_fb_px_m": self.stereo_fb,
            "stereo_roi_xy": [self.stereo_roi_x, self.stereo_roi_y],
        }


def save_prediction(
    output_directory: Path,
    result: dict[str, torch.Tensor],
    sample: SamplePaths,
    provenance: dict,
) -> dict:
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    depth = result["depth"].numpy()
    np.save(output_directory / "depth_m.npy", depth)
    torch.save(result["depth"], output_directory / "depth_m.pt")
    normalized = np.clip(depth / 100.0, 0.0, 1.0)
    colored_bgr = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
    )
    cv2.imwrite(str(output_directory / "depth_color.png"), colored_bgr)
    record = {
        "sample_id": sample.sample_id,
        "left": str(sample.left),
        "right": str(sample.right),
        "right_preprocessing": {
            "vertical_resize_hw": [
                EXPECTED_STEREO_SIZE[0],
                EXPECTED_RIGHT_SIZE[1],
            ],
            "interpolation": "cv2.INTER_LINEAR",
            "horizontal_phase_split": "phase::3",
            "generated_phase_hw": list(EXPECTED_STEREO_SIZE),
        },
        "sky_mask": str(sample.sky_mask) if sample.sky_mask is not None else None,
        "output_shape": list(depth.shape),
        "minimum_depth_m": float(depth.min()),
        "maximum_depth_m": float(depth.max()),
        "mean_depth_m": float(depth.mean()),
        "artifacts": {
            "metric_depth_numpy": "depth_m.npy",
            "metric_depth_torch": "depth_m.pt",
            "depth_color": "depth_color.png",
        },
        "provenance": provenance,
    }
    (output_directory / "prediction.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record
