"""Dataset for DQNet.

Consumes the NPZ shards produced by `dqnet.data.extract_features`. Each
training sample corresponds to a window of `1 + chunk_size` frames inside
a single demo:

    - frame t   : current observation (current visual + proprio)
    - frame t+1 .. t+chunk_size : future frames whose visual + actions we predict

Action labels for steps t .. t+chunk_size-1 (the actions taken to reach
those future frames). Visual labels are the future-frame DINOv2 CLS
tokens (agent + wrist), used for the cosine prediction loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleIndex:
    shard_idx: int
    demo_key: str
    start: int  # frame t


class LiberoChunkDataset(Dataset):
    """Loads chunked samples from cached DINOv2 feature shards."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        chunk_size: int = 6,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.stride = int(stride)
        self.shard_paths = [Path(p) for p in shard_paths]
        # We mmap each shard once and keep handles for cheap random access.
        self._shards: list[np.lib.npyio.NpzFile | None] = [None] * len(self.shard_paths)
        self._instructions: list[str] = []
        self._demo_keys: list[list[str]] = []

        self.samples: list[SampleIndex] = []
        for shard_idx, path in enumerate(self.shard_paths):
            with np.load(path, allow_pickle=True) as npz:
                instruction = str(npz["instruction"].item())
                demo_keys = [str(k) for k in npz["demo_keys"].tolist()]
                self._instructions.append(instruction)
                self._demo_keys.append(demo_keys)
                for demo_key in demo_keys:
                    actions = npz[f"{demo_key}/actions"]
                    T = actions.shape[0]
                    last_start = T - self.chunk_size - 1
                    if last_start < 0:
                        continue
                    for start in range(0, last_start + 1, self.stride):
                        self.samples.append(
                            SampleIndex(shard_idx=shard_idx, demo_key=demo_key, start=start)
                        )

        if not self.samples:
            raise RuntimeError("No samples produced — check shard paths / chunk_size")

    def __len__(self) -> int:
        return len(self.samples)

    def _shard(self, idx: int) -> np.lib.npyio.NpzFile:
        if self._shards[idx] is None:
            self._shards[idx] = np.load(self.shard_paths[idx], allow_pickle=True)
        return self._shards[idx]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        s = self.samples[i]
        shard = self._shard(s.shard_idx)
        d = s.demo_key
        agent = shard[f"{d}/agent_feat"]    # (T, D) float16
        wrist = shard[f"{d}/wrist_feat"]
        actions = shard[f"{d}/actions"]     # (T, 7) float32
        proprio = shard[f"{d}/proprio"]     # (T, 8) float32

        t = s.start
        k = self.chunk_size

        cur_agent = torch.from_numpy(agent[t].astype(np.float32))
        cur_wrist = torch.from_numpy(wrist[t].astype(np.float32))
        cur_proprio = torch.from_numpy(proprio[t])

        # Future targets: visuals are CLS tokens at t+1..t+k, actions are the
        # commanded actions a_t .. a_{t+k-1} that produce those frames.
        fut_agent = torch.from_numpy(agent[t + 1 : t + k + 1].astype(np.float32))
        fut_wrist = torch.from_numpy(wrist[t + 1 : t + k + 1].astype(np.float32))
        fut_actions = torch.from_numpy(actions[t : t + k])

        out: dict[str, torch.Tensor | str] = {
            "instruction": self._instructions[s.shard_idx],
            "cur_agent": cur_agent,           # (D,)
            "cur_wrist": cur_wrist,           # (D,)
            "cur_proprio": cur_proprio,       # (P,)
            "fut_agent": fut_agent,           # (K, D)
            "fut_wrist": fut_wrist,           # (K, D)
            "fut_actions": fut_actions,       # (K, 7)
        }

        # Optional supervised-target-token fields (method 3). Present only in
        # the target-feature cache; absent caches train the baseline unchanged.
        if f"{d}/agent_patch" in shard:
            out["cur_agent_patch"] = torch.from_numpy(
                shard[f"{d}/agent_patch"][t].astype(np.float32)
            )  # (N, D)
            out["cur_target_heatmap"] = torch.from_numpy(
                shard[f"{d}/agent_target_heatmap"][t].astype(np.float32)
            )  # (N,)
            valid = shard[f"{d}/target_valid"][t]
            out["cur_target_valid"] = torch.tensor(bool(valid), dtype=torch.bool)

        # Optional instruction embedding for instruction-aware target token.
        if "instruction_embed" in shard:
            out["instruction_embed"] = torch.from_numpy(
                shard["instruction_embed"].astype(np.float32)
            )  # (D,)

        return out


def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors and keep instructions as list of str (tokenized in model)."""
    out: dict[str, torch.Tensor | list[str]] = {}
    out["instruction"] = [b["instruction"] for b in batch]
    for key in [
        "cur_agent",
        "cur_wrist",
        "cur_proprio",
        "fut_agent",
        "fut_wrist",
        "fut_actions",
    ]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    # Optional target-token fields: stack only if every sample has them.
    for key in ["cur_agent_patch", "cur_target_heatmap", "cur_target_valid"]:
        if all(key in b for b in batch):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    # Instruction embed: keep as-is (same instruction for all samples in a batch).
    if "instruction_embed" in batch[0]:
        out["instruction_embed"] = torch.stack([b["instruction_embed"] for b in batch], dim=0)
    return out


def find_shards(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("*.npz"))
