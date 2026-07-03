"""Inference / rollout for a trained DQNet checkpoint.

Loads a checkpoint produced by `dqnet.train.save_checkpoint` (which stores only
the trainable weights: LoRA adapter + projections + heads) on top of a freshly
constructed model, then predicts an action chunk from a single observation.

The model consumes DINOv2 CLS features, not raw images, so this script also
provides a `FeatureEncoder` whose preprocessing matches
`dqnet.data.extract_features` exactly (vertical flip -> resize 224 -> normalize
-> CLS token). Use it when running on live env frames.

Quick check against cached features (no DINOv2 needed):
    python -m GlanceVLA.scripts.infer --ckpt output/run1/ckpt_final --from_cache

Live usage (encode raw RGB):
    encoder = FeatureEncoder(device=...)
    feat = encoder(agent_rgb_uint8)   # (B, 768)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import sys

sys.path.insert(0, ".")

from dqnet.models.dqnet import DQNet, DQNetConfig


def load_policy(ckpt_dir: str | Path, device: torch.device, dtype: torch.dtype) -> DQNet:
    """Rebuild the model and load the trainable weights from a checkpoint dir."""
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "config.yaml") as f:
        saved = yaml.safe_load(f)

    cfg = DQNetConfig(**saved)
    model = DQNet(cfg).to(device).to(dtype)

    state = torch.load(ckpt_dir / "trainable.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `missing` is expected: it is the frozen Qwen base (loaded via from_pretrained).
    # `unexpected` should be empty; warn if not, it means a name mismatch.
    loaded = len(state)
    print(f"[load] applied {loaded} trainable tensors from {ckpt_dir}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys in checkpoint: {unexpected[:5]} ...")
    model.eval()
    return model


class FeatureEncoder:
    """DINOv2 CLS extractor matching the offline preprocessing in extract_features."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        flip_vertical: bool = True,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.flip_vertical = flip_vertical
        processor = AutoImageProcessor.from_pretrained(model_name)
        self.mean = torch.tensor(processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, device=self.device).view(1, 3, 1, 1)
        self.model = AutoModel.from_pretrained(model_name).to(self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def __call__(self, images_uint8: np.ndarray) -> torch.Tensor:
        """images_uint8: (N, H, W, 3) uint8 -> (N, 768) float32 CLS tokens."""
        out = self._forward(images_uint8)
        return out.last_hidden_state[:, 0].float()  # (N, 768)

    @torch.no_grad()
    def patches(self, images_uint8: np.ndarray, grid_size: int = 8) -> torch.Tensor:
        """images_uint8: (N, H, W, 3) uint8 -> (N, grid_size**2, 768) float32.

        DINOv2-base on 224x224 yields a 16x16 patch grid (256 tokens). We
        avg-pool that grid down to grid_size x grid_size to keep the token
        budget (and feature cache) small. Must match extract_features_target.py.
        """
        out = self._forward(images_uint8)
        patch = out.last_hidden_state[:, 1:].float()  # (N, 256, D)
        n, p, d = patch.shape
        side = int(round(p ** 0.5))                   # 16
        if side * side != p:
            raise ValueError(f"non-square patch count {p}")
        grid = patch.permute(0, 2, 1).reshape(n, d, side, side)  # (N, D, 16, 16)
        if side != grid_size:
            grid = F.adaptive_avg_pool2d(grid, output_size=(grid_size, grid_size))
        grid = grid.reshape(n, d, grid_size * grid_size).permute(0, 2, 1)  # (N, G*G, D)
        return grid.contiguous()

    @torch.no_grad()
    def _forward(self, images_uint8: np.ndarray):
        x = images_uint8
        if self.flip_vertical:
            x = x[:, ::-1].copy()  # LIBERO frames are stored upside-down
        x = torch.from_numpy(x).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        return self.model(pixel_values=x)


def _demo_from_cache(model: DQNet, device: torch.device) -> None:
    """Sanity rollout using one window of pre-extracted cached features."""
    from dqnet.data.dataset import LiberoChunkDataset, find_shards, collate_fn
    from torch.utils.data import DataLoader

    ds = LiberoChunkDataset(find_shards("cache/dino_features"), chunk_size=model.cfg.chunk_size)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))

    actions = model.predict_action_chunk(
        instructions=batch["instruction"],
        cur_agent=batch["cur_agent"],
        cur_wrist=batch["cur_wrist"],
        cur_proprio=batch["cur_proprio"],
    )
    print(f"[infer] instruction: {batch['instruction'][0]!r}")
    print(f"[infer] predicted action chunk shape: {tuple(actions.shape)}")
    np.set_printoptions(precision=3, suppress=True)
    print(actions[0].float().cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="output/run1/ckpt_final")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--from_cache", action="store_true",
                        help="Run a sanity rollout on one cached feature window.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    model = load_policy(args.ckpt, device, dtype)

    if args.from_cache:
        _demo_from_cache(model, device)
    else:
        print("[infer] policy loaded. Use FeatureEncoder + model.predict_action_chunk "
              "on live frames, or pass --from_cache for a quick sanity check.")


if __name__ == "__main__":
    main()
