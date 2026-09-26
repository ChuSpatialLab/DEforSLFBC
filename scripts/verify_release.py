#!/usr/bin/env python3
"""Audit the self-contained release, with optional model and inference checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXPECTED_SAMPLE_COUNT = 100

from deforslfbc.pipeline import preprocess_right_image  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the full JSON report; stdout is always printed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = []

    def check(name: str, condition: bool, detail=None):
        record = {"name": name, "passed": bool(condition)}
        if detail is not None:
            record["detail"] = detail
        checks.append(record)
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    expected_root_files = {
        ".gitattributes",
        ".gitignore",
        "README.md",
        "THIRD_PARTY.md",
        "environment.yml",
        "requirements.txt",
        "infer.py",
        "evaluate.py",
    }
    check(
        "required root files",
        all((ROOT / name).is_file() for name in expected_root_files),
        sorted(name for name in expected_root_files if not (ROOT / name).is_file()),
    )

    expected_readme_assets = {
        "acquisition-platform.png": (1278, 1940),
        "captured-image-pair.png": (854, 2566),
        "rectified-image-pair.png": (914, 2452),
        "inference-example-000030.png": (1660, 1280),
        "result-example-000030.png": (1660, 1280),
    }
    asset_root = ROOT / "docs" / "assets"
    for name, expected_shape in expected_readme_assets.items():
        asset = cv2.imread(str(asset_root / name), cv2.IMREAD_UNCHANGED)
        check(
            f"README asset {name}",
            asset is not None and asset.shape[:2] == expected_shape,
            None if asset is None else list(asset.shape),
        )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check(
        "README uses permanent assets",
        all(f'docs/assets/{name}' in readme for name in expected_readme_assets)
        and "reference/" not in readme,
    )

    checkpoint_root = ROOT / "checkpoints"
    checkpoint_manifest = json.loads(
        (checkpoint_root / "manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_paths = sorted(checkpoint_root.glob("*.pth"))
    check(
        "exactly three checkpoints",
        [path.name for path in checkpoint_paths]
        == ["fusion.pth", "mono.pth", "stereo.pth"],
        [path.name for path in checkpoint_paths],
    )
    expected_formats = {
        "mono.pth": "deforslfbc-mono-inference-v1",
        "stereo.pth": "deforslfbc-stereo-inference-v1",
        "fusion.pth": "deforslfbc-fusion-inference-v1",
    }
    forbidden_training_keys = {
        "optimizer",
        "optimizer_state_dict",
        "scheduler",
        "scheduler_state_dict",
        "scaler",
    }
    for checkpoint_path in checkpoint_paths:
        record = checkpoint_manifest["files"][checkpoint_path.name]
        check(
            f"{checkpoint_path.name} byte size",
            checkpoint_path.stat().st_size == int(record["bytes"]),
            checkpoint_path.stat().st_size,
        )
        if not args.skip_file_hashes:
            check(
                f"{checkpoint_path.name} SHA-256",
                sha256_file(checkpoint_path) == record["sha256"],
                record["sha256"],
            )
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        check(
            f"{checkpoint_path.name} format",
            payload.get("format") == expected_formats[checkpoint_path.name],
            payload.get("format"),
        )
        check(
            f"{checkpoint_path.name} inference-only",
            forbidden_training_keys.isdisjoint(payload),
            sorted(forbidden_training_keys.intersection(payload)),
        )

    data_root = ROOT / "test_data"
    data_manifest = json.loads(
        (data_root / "manifest.json").read_text(encoding="utf-8")
    )
    samples = data_manifest["samples"]
    check("manifest schema", data_manifest.get("schema_version") == 3)
    check(
        "manifest sample count",
        data_manifest["sample_count"] == EXPECTED_SAMPLE_COUNT,
        data_manifest["sample_count"],
    )
    check(
        "actual sample count",
        len(samples) == EXPECTED_SAMPLE_COUNT,
        len(samples),
    )
    check("manifest right size", data_manifest.get("right_size_wh") == [1920, 1152])
    check(
        "manifest right preprocessing",
        data_manifest.get("right_preprocessing")
        == {
            "vertical_resize_size_wh": [1920, 384],
            "interpolation": "cv2.INTER_LINEAR",
            "horizontal_phase_split": "phase::3",
            "phase_count": 3,
            "generated_phase_size_wh": [640, 384],
        },
        data_manifest.get("right_preprocessing"),
    )
    check(
        "manifest sky mask",
        data_manifest.get("sky_mask")
        == {
            "size_wh": [1280, 768],
            "dtype": "uint8",
            "values": [0, 255],
            "alignment": "left full frame",
        },
        data_manifest.get("sky_mask"),
    )
    sample_ids = [record["sample_id"] for record in samples]
    check(
        "unique sample IDs",
        len(set(sample_ids)) == EXPECTED_SAMPLE_COUNT,
    )
    expected_directories = {
        f"{index:06d}" for index in range(EXPECTED_SAMPLE_COUNT)
    }
    check(
        "merged sequential directories",
        {record["directory"] for record in samples} == expected_directories,
    )
    check(
        "date-free release identifiers",
        all(
            record["sample_id"] == record["directory"]
            and set(record).isdisjoint({"source_sequence", "source_filename"})
            for record in samples
        )
        and "sequences" not in data_manifest,
    )
    selection = data_manifest.get("selection", {})
    check(
        "diversity selection metadata",
        selection.get("candidate_count") == 200
        and selection.get("retained_count") == EXPECTED_SAMPLE_COUNT
        and selection.get("adjacent_frames_retained") is False,
        selection,
    )
    check(
        "merged layout metadata",
        data_manifest.get("layout")
        == {
            "type": "merged sequential sample directories",
            "directory_pattern": "000000-000099",
            "collection_date_directories": False,
        },
        data_manifest.get("layout"),
    )
    expected_names = {
        "left.png",
        "right.png",
        "ground_truth.png",
        "sky_mask.png",
    }
    left_hashes = []
    right_hashes = []
    for record in samples:
        directory = data_root / record["directory"]
        check(
            f"{record['sample_id']} directory",
            directory.is_dir(),
            str(directory.relative_to(ROOT)),
        )
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        check(
            f"{record['sample_id']} exact files",
            actual_names == expected_names,
            sorted(actual_names),
        )
        for name in sorted(expected_names):
            path = directory / name
            manifest_file = record["files"][name]
            check(
                f"{record['sample_id']} {name} bytes",
                path.stat().st_size == int(manifest_file["bytes"]),
            )
            if not args.skip_file_hashes:
                check(
                    f"{record['sample_id']} {name} SHA-256",
                    sha256_file(path) == manifest_file["sha256"],
                )
        left_hashes.append(record["files"]["left.png"]["sha256"])
        right_hashes.append(record["files"]["right.png"]["sha256"])
        left = cv2.imread(str(directory / "left.png"), cv2.IMREAD_COLOR)
        check(
            f"{record['sample_id']} left shape",
            left is not None and left.shape[:2] == (768, 1280),
            None if left is None else list(left.shape),
        )
        right = np.asarray(
            Image.open(directory / "right.png").convert("RGB"), dtype=np.uint8
        )
        check(
            f"{record['sample_id']} right shape",
            right.shape == (1152, 1920, 3),
            list(right.shape),
        )
        right_phases = preprocess_right_image(right)
        expected_phase_hashes = record["derived_right_phase_pixel_sha256"]
        check(
            f"{record['sample_id']} derived phase count",
            len(right_phases) == len(expected_phase_hashes) == 3,
        )
        for phase, (right_phase, expected_hash) in enumerate(
            zip(right_phases, expected_phase_hashes)
        ):
            check(
                f"{record['sample_id']} derived right phase {phase} shape",
                right_phase.shape == (384, 640, 3),
                list(right_phase.shape),
            )
            check(
                f"{record['sample_id']} derived right phase {phase} pixels",
                hashlib.sha256(right_phase.tobytes()).hexdigest() == expected_hash,
            )
        ground_truth = cv2.imread(
            str(directory / "ground_truth.png"), cv2.IMREAD_UNCHANGED
        )
        check(
            f"{record['sample_id']} GT",
            ground_truth is not None
            and ground_truth.dtype == np.uint16
            and ground_truth.shape == (768, 1280),
            None
            if ground_truth is None
            else {"shape": list(ground_truth.shape), "dtype": str(ground_truth.dtype)},
        )
        sky_mask = cv2.imread(
            str(directory / "sky_mask.png"), cv2.IMREAD_UNCHANGED
        )
        sky_values = set() if sky_mask is None else set(np.unique(sky_mask).tolist())
        check(
            f"{record['sample_id']} sky mask",
            sky_mask is not None
            and sky_mask.dtype == np.uint8
            and sky_mask.shape == (768, 1280)
            and sky_values.issubset({0, 255}),
            None
            if sky_mask is None
            else {
                "shape": list(sky_mask.shape),
                "dtype": str(sky_mask.dtype),
                "values": sorted(sky_values),
            },
        )
    check(
        "all left images unique",
        len(set(left_hashes)) == EXPECTED_SAMPLE_COUNT,
    )
    check(
        "all right images unique",
        len(set(right_hashes)) == EXPECTED_SAMPLE_COUNT,
    )

    text_extensions = {".py", ".md", ".yml", ".yaml", ".json", ".txt"}
    forbidden_fragments = ("/home/", "/mnt/", "/media/")
    leaked_paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                leaked_paths.append(f"{path.relative_to(ROOT)}:{fragment}")
    check("no machine-specific absolute paths", not leaked_paths, leaked_paths)

    model_record = None
    inference_record = None
    if args.load_models or args.run_inference:
        from deforslfbc.pipeline import DepthFusionPipeline, SamplePaths

        pipeline = DepthFusionPipeline(checkpoint_root, device=args.device)
        model_record = {
            "device": str(pipeline.device),
            "stereo_fb_px_m": pipeline.stereo_fb,
            "stereo_roi_xy": [pipeline.stereo_roi_x, pipeline.stereo_roi_y],
            "max_depth_m": pipeline.max_depth,
        }
        check(
            "checkpoint stereo FB",
            abs(pipeline.stereo_fb - 146.98806411139032) < 1e-9,
            pipeline.stereo_fb,
        )
        check(
            "checkpoint stereo ROI",
            (pipeline.stereo_roi_x, pipeline.stereo_roi_y) == (320, 192),
        )
        if args.run_inference:
            sample = SamplePaths.from_directory(
                data_root / samples[0]["directory"]
            )
            result = pipeline.predict(sample)
            depth = result["depth"].numpy()
            target_u16 = cv2.imread(
                str(sample.ground_truth), cv2.IMREAD_UNCHANGED
            )
            target = target_u16.astype(np.float32) * 300.0 / 65535.0
            valid = (target > 0.0) & (target < 100.0)
            abs_rel = float(
                np.mean(np.abs(np.maximum(depth[valid], 0.01) - target[valid]) / target[valid])
            )
            inference_record = {
                "sample_id": sample.sample_id,
                "shape": list(depth.shape),
                "all_finite": bool(np.isfinite(depth).all()),
                "minimum_depth_m": float(depth.min()),
                "maximum_depth_m": float(depth.max()),
                "abs_rel": abs_rel,
            }
            check("end-to-end output shape", depth.shape == (768, 1280))
            check("end-to-end output finite", np.isfinite(depth).all())
            check("end-to-end output positive", float(depth.min()) > 0.0)
            check("end-to-end output max depth", float(depth.max()) <= 100.0)

    report = {
        "status": "passed",
        "check_count": len(checks),
        "all_passed": all(record["passed"] for record in checks),
        "model_load": model_record,
        "end_to_end_inference": inference_record,
        "checks": checks,
    }
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "check_count": report["check_count"],
                "all_passed": report["all_passed"],
                "model_load": model_record,
                "end_to_end_inference": inference_record,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
