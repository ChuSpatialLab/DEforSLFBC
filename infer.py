#!/usr/bin/env python3
"""Run one end-to-end DEforSLFBC prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from deforslfbc.pipeline import (  # noqa: E402
    DepthFusionPipeline,
    SamplePaths,
    save_prediction,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        help="Directory containing left.png and right.png",
    )
    parser.add_argument("--left", type=Path, help="Full-frame left RGB image")
    parser.add_argument(
        "--right",
        type=Path,
        help="Full rectified right RGB image (1152x1920)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.sample_dir is not None:
        if args.left is not None or args.right is not None:
            parser.error("--sample-dir cannot be combined with --left/--right")
    elif args.left is None or args.right is None:
        parser.error("provide either --sample-dir or both --left and --right")
    return args


def main() -> None:
    args = parse_args()
    if args.sample_dir is not None:
        sample = SamplePaths.from_directory(args.sample_dir)
    else:
        sample = SamplePaths(
            left=args.left.expanduser().resolve(),
            right=args.right.expanduser().resolve(),
            sample_id=args.left.stem,
        )
    pipeline = DepthFusionPipeline(args.checkpoints, device=args.device)
    result = pipeline.predict(sample)
    record = save_prediction(
        args.output, result, sample, pipeline.checkpoint_provenance()
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
