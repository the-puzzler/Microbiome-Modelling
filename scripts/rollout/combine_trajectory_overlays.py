#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GINGIVA_PNG = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay.png")
DEFAULT_DIABIMMUNE_PNG = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectory_overlay.png")
DEFAULT_OUT_PNG = os.path.join("data", "rollout", "trajectory_overlays_side_by_side.png")


def parse_args():
    p = argparse.ArgumentParser(description="Combine gingivitis + diabimmune trajectory overlay PNGs side-by-side.")
    p.add_argument("--gingiva", default=DEFAULT_GINGIVA_PNG, help="Path to gingivitis overlay PNG.")
    p.add_argument("--diabimmune", default=DEFAULT_DIABIMMUNE_PNG, help="Path to diabimmune overlay PNG.")
    p.add_argument("--out", default=DEFAULT_OUT_PNG, help="Output PNG path.")
    p.add_argument("--pad", type=int, default=30, help="Horizontal padding (pixels) between panels.")
    return p.parse_args()


def to_rgba(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 3:
        alpha = np.ones((*img.shape[:-1], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=-1)
    if img.shape[-1] != 4:
        raise ValueError(f"Expected 3 or 4 channels, got shape {img.shape}")
    return img


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad_total = height - img.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_width = ((pad_top, pad_bottom), (0, 0), (0, 0))
    white = 1.0 if np.issubdtype(img.dtype, np.floating) else 255
    return np.pad(img, pad_width, mode="constant", constant_values=white)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not os.path.exists(args.gingiva):
        raise SystemExit(f"Missing gingivitis PNG: {args.gingiva}")
    if not os.path.exists(args.diabimmune):
        raise SystemExit(f"Missing diabimmune PNG: {args.diabimmune}")

    left = to_rgba(plt.imread(args.gingiva))
    right = to_rgba(plt.imread(args.diabimmune))

    height = max(left.shape[0], right.shape[0])
    left = pad_to_height(left, height)
    right = pad_to_height(right, height)

    white = 1.0 if np.issubdtype(left.dtype, np.floating) else 255
    pad = np.full((height, max(0, int(args.pad)), 4), fill_value=white, dtype=left.dtype)
    combined = np.concatenate([left, pad, right], axis=1)

    plt.imsave(args.out, combined)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

