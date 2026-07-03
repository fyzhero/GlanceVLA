"""Pre-extract DINOv2 CLS tokens for every frame in LIBERO HDF5 demos.

Reads each demo's `agentview_rgb` and `eye_in_hand_rgb` (both uint8 HxWx3
at 128x128), resizes to 224x224, runs DINOv2-base, and stores per-demo
CLS tokens together with the original action/proprio data into a compact
NPZ shard. The shards are then consumed by `dqnet.data.dataset.LiberoChunkDataset`.

Run on a small subset first:
    python -m dqnet.data.extract_features --max_files 1

Run for everything:
    python -m dqnet.data.extract_features
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def list_hdf5_files(roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            print(f"[warn] dataset dir missing: {root_p}")
            continue
        files.extend(sorted(root_p.glob("*.hdf5")))
    return files


def task_name_from_path(p: Path) -> str:
    """Strip the trailing `_demo` and replace underscores with spaces."""
    stem = p.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")


@torch.no_grad()
def encode_images_batch(
    images_uint8: np.ndarray,  # (N, H, W, 3) uint8
    model: torch.nn.Module,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Run DINOv2 over a batch of frames and return float16 CLS tokens (N, D)."""
    n = images_uint8.shape[0]
    out_chunks: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        batch = images_uint8[i : i + batch_size]
        x = torch.from_numpy(batch).to(device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - image_mean) / image_std
        x = x.to(dtype)
        out = model(pixel_values=x)
        cls = out.last_hidden_state[:, 0]  # (B, D)
        out_chunks.append(cls.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out_chunks, axis=0)


def process_file(
    h5_path: Path,
    cache_dir: Path,
    model: torch.nn.Module,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    overwrite: bool,
) -> int:
    """Extract features for one HDF5 file. Returns number of demos written."""
    out_path = cache_dir / f"{h5_path.stem}.npz"
    if out_path.exists() and not overwrite:
        return 0

    image_mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)

    instruction = task_name_from_path(h5_path)
    payload: dict[str, np.ndarray] = {}
    n_demos = 0

    with h5py.File(h5_path, "r") as f:
        demo_keys = sorted(
            f["data"].keys(), key=lambda k: int(k.split("_")[-1])
        )
        for demo_key in tqdm(demo_keys, desc=h5_path.stem, leave=False):
            grp = f[f"data/{demo_key}"]
            agent = grp["obs/agentview_rgb"][:]      # (T, 128, 128, 3) uint8
            wrist = grp["obs/eye_in_hand_rgb"][:]
            actions = grp["actions"][:].astype(np.float32)         # (T, 7)
            ee_pos = grp["obs/ee_pos"][:].astype(np.float32)       # (T, 3)
            ee_ori = grp["obs/ee_ori"][:].astype(np.float32)       # (T, 3)
            grip = grp["obs/gripper_states"][:].astype(np.float32) # (T, 2)
            proprio = np.concatenate([ee_pos, ee_ori, grip], axis=-1)  # (T, 8)

            # LIBERO images are stored upside-down in the HDF5; flip vertically
            # before feeding the encoder so they match what robosuite renders.
            agent = agent[:, ::-1].copy()
            wrist = wrist[:, ::-1].copy()

            agent_feat = encode_images_batch(
                agent, model, image_mean, image_std, batch_size, device, dtype
            )
            wrist_feat = encode_images_batch(
                wrist, model, image_mean, image_std, batch_size, device, dtype
            )

            payload[f"{demo_key}/agent_feat"] = agent_feat
            payload[f"{demo_key}/wrist_feat"] = wrist_feat
            payload[f"{demo_key}/actions"] = actions
            payload[f"{demo_key}/proprio"] = proprio
            n_demos += 1

    payload["instruction"] = np.array(instruction)
    payload["demo_keys"] = np.array(
        sorted(set(k.split("/")[0] for k in payload if "/" in k))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return n_demos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=["datasets/libero_object", "datasets/libero_spatial"],
    )
    parser.add_argument("--cache_dir", default="cache/dino_features")
    parser.add_argument("--model_name", default="facebook/dinov2-base")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[info] loading {args.model_name} on {device}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device, dtype=dtype).eval()

    files = list_hdf5_files(args.dataset_dirs)
    if args.max_files is not None:
        files = files[: args.max_files]
    print(f"[info] {len(files)} hdf5 files to process")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_demos = 0
    for h5_path in files:
        n = process_file(
            h5_path,
            cache_dir,
            model,
            processor,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        total_demos += n
        print(f"[info] {h5_path.name}: +{n} demos -> {cache_dir}")

    print(f"[done] processed {total_demos} demos across {len(files)} files")


if __name__ == "__main__":
    main()
