#!/usr/bin/env python3
"""
Scan PNG images in a folder and report which files look blurry.

Blur score is the variance of the Laplacian:
- lower score = blurrier image
- image is marked BLUR if score < threshold
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(
        "Missing dependency. Install OpenCV first:\n"
        "  pip install opencv-python numpy",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


@dataclass(frozen=True)
class BlurResult:
    path: Path
    score: float
    is_blurry: bool


def load_image_gray(path: Path) -> np.ndarray | None:
    """Load image as grayscale. This supports Unicode paths better on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None

    if data.size == 0:
        return None

    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return image


def blur_score(gray_image: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def iter_png_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def check_folder(folder: Path, threshold: float, recursive: bool) -> tuple[list[BlurResult], list[Path]]:
    results: list[BlurResult] = []
    failed: list[Path] = []

    for path in iter_png_files(folder, recursive):
        image = load_image_gray(path)
        if image is None:
            failed.append(path)
            continue

        score = blur_score(image)
        results.append(BlurResult(path=path, score=score, is_blurry=score < threshold))

    results.sort(key=lambda item: item.score)
    return results, failed


def write_csv(path: Path, results: list[BlurResult], failed: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["status", "score", "path"])
        for item in results:
            writer.writerow(["BLUR" if item.is_blurry else "OK", f"{item.score:.2f}", str(item.path)])
        for item in failed:
            writer.writerow(["READ_ERROR", "", str(item)])


def print_report(results: list[BlurResult], failed: list[Path], threshold: float) -> None:
    blurry_count = sum(1 for item in results if item.is_blurry)

    print(f"Threshold: {threshold:.2f}")
    print(f"Checked: {len(results)} PNG files")
    print(f"Blurry: {blurry_count}")
    if failed:
        print(f"Read error: {len(failed)}")
    print()

    if not results and not failed:
        print("No PNG files found.")
        return

    print(f"{'STATUS':<10} {'SCORE':>12}  PATH")
    print("-" * 80)
    for item in results:
        status = "BLUR" if item.is_blurry else "OK"
        print(f"{status:<10} {item.score:>12.2f}  {item.path}")

    for path in failed:
        print(f"{'READ_ERROR':<10} {'':>12}  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check PNG images in a folder and report blurry images.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder that contains PNG images.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Images with score below this value are marked BLUR. Default: 100.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan PNG files directly inside the folder.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV report output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder

    if not folder.exists():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2
    if not folder.is_dir():
        print(f"Path is not a folder: {folder}", file=sys.stderr)
        return 2
    if args.threshold <= 0:
        print("Threshold must be greater than 0.", file=sys.stderr)
        return 2

    results, failed = check_folder(
        folder=folder,
        threshold=args.threshold,
        recursive=not args.no_recursive,
    )
    print_report(results, failed, args.threshold)

    if args.csv:
        write_csv(args.csv, results, failed)
        print(f"\nCSV saved: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
