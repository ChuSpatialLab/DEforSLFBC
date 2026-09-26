#!/usr/bin/env python3
"""Run end-to-end inference and evaluate the bundled outdoor test subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from deforslfbc.pipeline import (  # noqa: E402
    DepthFusionPipeline,
    SamplePaths,
    save_prediction,
)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(target) & (target > 0.0) & (target < 100.0)
    if not valid.any():
        raise RuntimeError("Ground truth contains no valid 0-100 m pixels")
    prediction = np.maximum(prediction[valid], 0.01)
    target = target[valid]
    difference = np.abs(prediction - target)
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "abs_rel": float(np.mean(difference / target)),
        "sq_rel": float(np.mean(np.square(difference) / target)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "rmse_log": float(
            np.sqrt(np.mean(np.square(np.log(prediction) - np.log(target))))
        ),
        "a1": float(np.mean(ratio < 1.25)),
        "a2": float(np.mean(ratio < 1.25**2)),
        "valid_pixels": int(valid.sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "test_data")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Only retain metrics.json, not per-frame depth outputs",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit]
    pipeline = DepthFusionPipeline(args.checkpoints, device=args.device)
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    per_frame = []
    started = time.time()
    for index, record in enumerate(samples, 1):
        sample_directory = data_root / record["directory"]
        sample = SamplePaths.from_directory(sample_directory)
        result = pipeline.predict(sample)
        ground_truth_u16 = cv2.imread(
            str(sample.ground_truth), cv2.IMREAD_UNCHANGED
        )
        if ground_truth_u16 is None or ground_truth_u16.dtype != np.uint16:
            raise RuntimeError(f"Invalid ground truth: {sample.ground_truth}")
        target = ground_truth_u16.astype(np.float32) * 300.0 / 65535.0
        frame_metrics = metrics(result["depth"].numpy(), target)
        frame_record = {
            "sample_id": sample.sample_id,
            **frame_metrics,
        }
        per_frame.append(frame_record)
        if not args.no_save_predictions:
            save_prediction(
                output_root / sample.sample_id,
                result,
                sample,
                pipeline.checkpoint_provenance(),
            )
        print(
            f"[{index}/{len(samples)}] {sample.sample_id}: "
            f"AbsRel={frame_metrics['abs_rel']:.6f}"
        )
    metric_names = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2")
    summary = {
        "samples": len(per_frame),
        "elapsed_seconds": time.time() - started,
        "frame_averaged": {
            name: float(np.mean([record[name] for record in per_frame]))
            for name in metric_names
        },
        "checkpoint_provenance": pipeline.checkpoint_provenance(),
        "per_frame": per_frame,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["frame_averaged"], indent=2))


if __name__ == "__main__":
    main()
