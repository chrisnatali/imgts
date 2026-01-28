#!/usr/bin/env python3
"""
Compute greenness statistic time series from a table of image paths + timestamps, using a static bitmask.

Input (--image_table_path):
  - CSV or Parquet with columns: image_path, timestamp
    - image_path: path to JPEG/PNG image (all images should be same dimensions)
    - timestamp: timestamp associated with corresponding image


Mask (--mask_path):
  - Bitmask image (aka Region of Interest [ROI]) of the same dimensions as the timeseries images
  - Applied to all images before processing to select pixels of interest
  - Nonzero pixels are interpreted as True.

Output (--output_path) Where to write the output statistic CSV with columns:
  timestamp, image_path, n_masked_pixels, mean_r, mean_g, mean_b, mean_exg, mean_g_over_sum, mean_g_minus_r

Performance:
  - Per-pixel math is batched, using array/operator broadcasting to process entire batch as a matrix and is
    GPU-friendly (CUDA if available).

Python: 3.13+
"""

from __future__ import annotations
from . import io

import argparse
from pathlib import Path

import csv
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import get_worker_info


class ImageTable(Dataset):
    """
    Torch Dataset representing
    Loads image paths + timestamps from a dataframe-like table.

    We keep timestamps as strings exactly as provided (no parsing) to avoid surprises.
    """

    def __init__(self, table: pd.DataFrame, expected_hw: tuple[int, int]):
        if "image_path" not in table.columns or "timestamp" not in table.columns:
            raise ValueError("Input table must contain columns: image_path, timestamp")

        self.image_paths = table["image_path"].astype(str).tolist()
        self.timestamps = table["timestamp"].astype(str).tolist()
        self.expected_hw = expected_hw  # (H, W)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        ts = self.timestamps[idx]

        wi = get_worker_info()
        wid = wi.id if wi else "main"

        try:
            rgb_hwc = io.load_image(img_path)  # uint8 (H, W, 3)

        except Exception as e:
            print(
                f"[worker {wid}] failed idx={idx} img_path={img_path}: {e}", flush=True
            )
            raise

        if rgb_hwc.shape[:2] != self.expected_hw:
            raise ValueError(
                f"Image size {rgb_hwc.shape[:2]} != mask size {self.expected_hw} for {img_path}"
            )

        return {"image_path": img_path, "timestamp": ts, "rgb_hwc": rgb_hwc}


def collate(batch: list[dict]) -> dict:
    """
    Stacks images into a float32 tensor (N, 3, H, W).
    Carries through timestamp + image_path as lists.
    """
    image_paths = [b["image_path"] for b in batch]
    timestamps = [b["timestamp"] for b in batch]

    rgbs = np.stack([b["rgb_hwc"] for b in batch], axis=0)  # (N, H, W, 3) uint8
    rgb_t = (
        # permute to (N,3,H,W) matrix
        # make these dimensions contiguous in memory for efficiency
        # cast to float32 here to avoid float conversions during arithmetic computations
        torch.from_numpy(rgbs).permute(0, 3, 1, 2).contiguous().to(torch.float32)
    )

    return {"image_paths": image_paths, "timestamps": timestamps, "rgb": rgb_t}


@torch.inference_mode()
def masked_metrics_batch(
    rgb_nchw: torch.Tensor, mask_hw: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    ---------------------------
    Batched masked metrics (Uses broadcasting over tensor and is GPU-friendly for performance)

    FUTURE: This can be a custom function written per specific image analysis goal
    ---------------------------

    # Dimension descriptions as NCHW or (N, C, H, W)
    #
    # - **N** = Index within batch
    # - **C** = Channel (e.g., color channels)
    # - **H** = Height (vertical spatial dimension, rows, Y-coordinate)
    # - **W** = Width (horizontal spatial dimension, columns, X-coordinate)

    rgb_nchw: float32, (N, 3, H, W), values 0..255
    mask_hw:  bool,     (H, W), True = include pixel

    Returns dict of tensors, each shape (N,).
    """
    # Expand mask to (1,1,H,W) so it broadcasts across batch and channel dims when
    # multiplied into an (N,C,H,W) image tensor. Keep a float copy for arithmetic.
    m = mask_hw.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    m_f = m.to(rgb_nchw.dtype)  # cast mask to same dtype as images (float32)

    # Included pixel count (same for every sample; static mask). Clamp to >=1 to avoid
    # division-by-zero later, then convert to float so divisions produce float outputs.
    n = mask_hw.sum().clamp_min(1).to(rgb_nchw.dtype)  # scalar (0-D tensor)

    # Slice out channels but keep channel dimension as a singleton (N,1,H,W). This
    # shape makes subsequent masked sums return shape (N,1) before squeezing.
    r = rgb_nchw[:, 0:1]
    g = rgb_nchw[:, 1:2]
    b = rgb_nchw[:, 2:3]

    # Compute masked sums per-channel over spatial dims (H,W). Result after sum is
    # shape (N,1); squeeze to (N,).
    sum_r = (r * m_f).sum(dim=(2, 3)).squeeze(1)
    sum_g = (g * m_f).sum(dim=(2, 3)).squeeze(1)
    sum_b = (b * m_f).sum(dim=(2, 3)).squeeze(1)

    # Per-image means over included pixels. n is scalar float (same for every image
    # since the mask is static); broadcasting yields shape (N,).
    mean_r = sum_r / n
    mean_g = sum_g / n
    mean_b = sum_b / n

    # Excess green per-pixel then mean over mask. Uses float dtype so negatives are
    # represented correctly (uint8 would wrap/clip).
    exg = 2.0 * g - r - b
    mean_exg = ((exg * m_f).sum(dim=(2, 3)).squeeze(1)) / n

    # Compute green fraction per-pixel, clamp the denominator to avoid div-by-zero,
    # then average over masked pixels.
    denom = (r + g + b).clamp_min(1e-6)
    g_over_sum = g / denom
    mean_g_over_sum = ((g_over_sum * m_f).sum(dim=(2, 3)).squeeze(1)) / n

    # Mean of green minus red over masked pixels.
    g_minus_r = g - r
    mean_g_minus_r = ((g_minus_r * m_f).sum(dim=(2, 3)).squeeze(1)) / n

    return {
        "n_masked_pixels": n.expand(rgb_nchw.shape[0]),
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_exg": mean_exg,
        "mean_g_over_sum": mean_g_over_sum,
        "mean_g_minus_r": mean_g_minus_r,
    }


@torch.inference_mode()
def run(
    images_table_path: Path,
    mask_path: Path,
    out_csv: Path,
    batch_size: int,
    workers: int,
) -> None:
    """
    This is the main data processing pipeline flow that
    1. gathers the static info (image paths and timestamps, and bit mask)
    2. collates the images into batches
    3. processes each batch, computing the metrics per image row within the batch
    4. outputs the table of metrics
    """

    table = io.read_image_table(images_table_path)
    if len(table) == 0:
        raise RuntimeError(f"No rows found in {images_table_path}")

    mask_bool_hw = io.load_image_mask(mask_path)
    expected_hw = mask_bool_hw.shape  # (H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Move static mask to device ONCE.
    mask_hw_t = torch.from_numpy(mask_bool_hw).to(torch.bool)
    mask_hw_t = mask_hw_t.to(device, non_blocking=use_cuda)

    ds = ImageTable(table, expected_hw)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # preserve input order
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=(workers > 0),
        collate_fn=collate,
    )

    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    fieldnames = [
        "timestamp",
        "image_path",
        "n_masked_pixels",
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_exg",
        "mean_g_over_sum",
        "mean_g_minus_r",
    ]

    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    batch_num = 0
    for batch in loader:
        rows: list[dict[str, object]] = []
        rgb = batch["rgb"].to(device, non_blocking=use_cuda)

        metrics = masked_metrics_batch(rgb, mask_hw_t)
        metrics_cpu = {k: v.detach().cpu().numpy() for k, v in metrics.items()}
        for i in range(len(batch["image_paths"])):
            rows.append(
                {
                    # TODO: Along with masked_metrics_batch, the output rows should be customized to match to allow the
                    # same processing framework to support varying statistical analyses
                    "timestamp": batch["timestamps"][i],
                    "image_path": batch["image_paths"][i],
                    "n_masked_pixels": float(metrics_cpu["n_masked_pixels"][i]),
                    "mean_r": float(metrics_cpu["mean_r"][i]),
                    "mean_g": float(metrics_cpu["mean_g"][i]),
                    "mean_b": float(metrics_cpu["mean_b"][i]),
                    "mean_exg": float(metrics_cpu["mean_exg"][i]),
                    "mean_g_over_sum": float(metrics_cpu["mean_g_over_sum"][i]),
                    "mean_g_minus_r": float(metrics_cpu["mean_g_minus_r"][i]),
                }
            )

        csv_writer.writerows(rows)
        print(f"Finished batch num {batch_num} with {len(rows)} rows")
        batch_num = batch_num + 1

    csv_file.close()
    print(f"Device used: {device}")
    print(f"Mask included pixels: {int(mask_bool_hw.sum())} / {mask_bool_hw.size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images_table_path",
        type=Path,
        required=True,
        help="CSV/Parquet with columns image_path,timestamp",
    )
    ap.add_argument(
        "--mask_path", type=Path, required=True, help="Bitmask image (nonzero=include)"
    )
    ap.add_argument("--output_path", type=Path, required=True, help="Output CSV path")
    ap.add_argument(
        "--batch", type=int, default=32, help="Number of images processed per batch"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="(EXPERIMENTAL) Number of subprocesses to process images with",
    )
    args = ap.parse_args()

    run(
        args.images_table_path,
        args.mask_path,
        args.output_path,
        args.batch,
        args.workers,
    )


if __name__ == "__main__":
    main()
