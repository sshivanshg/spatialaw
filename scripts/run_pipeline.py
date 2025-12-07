#!/usr/bin/env python3
"""
Convenience CLI to orchestrate the full WiAR processing + training pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW = ROOT / "data" / "raw" / "WiAR"
DEFAULT_WINDOWS = ROOT / "data" / "processed" / "windows"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "features"
DEFAULT_BINARY = ROOT / "data" / "processed" / "binary"
DEFAULT_MODELS = ROOT / "models"

ALL_STEPS = ["windows", "features", "binary", "validate", "tune"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Subset of steps to run (default: run all).",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--windows-dir", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--binary-dir", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--T", type=int, default=256, help="Window length.")
    parser.add_argument("--stride", type=int, default=64, help="Window stride.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--motion-quantile", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print(f"\n→ Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    steps = args.steps

    if "windows" in steps:
        run(
            [
                sys.executable,
                "scripts/generate_windows.py",
                "--input-dir",
                str(args.raw_dir),
                "--out-dir",
                str(args.windows_dir),
                "--T",
                str(args.T),
                "--stride",
                str(args.stride),
                "--seed",
                str(args.seed),
            ]
        )

    if "features" in steps:
        run(
            [
                sys.executable,
                "scripts/extract_features.py",
                "--windows-dir",
                str(args.windows_dir),
                "--output-dir",
                str(args.features_dir),
                "--seed",
                str(args.seed),
            ]
        )

    if "binary" in steps:
        run(
            [
                sys.executable,
                "scripts/process_binary_dataset.py",
                "--wiar-features-dir",
                str(args.features_dir),
                "--out-dir",
                str(args.binary_dir),
                "--motion-quantile",
                str(args.motion_quantile),
            ]
        )

    if "validate" in steps:
        run(
            [
                sys.executable,
                "scripts/validate_binary_dataset.py",
                "--binary-dir",
                str(args.binary_dir),
            ]
        )

    if "tune" in steps:
        run(
            [
                sys.executable,
                "model_tools/tune_presence_detector.py",
                "--binary-dir",
                str(args.binary_dir),
                "--models-dir",
                str(args.models_dir),
                "--cv-folds",
                str(args.cv_folds),
            ]
        )

    print("\n✓ Pipeline complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

