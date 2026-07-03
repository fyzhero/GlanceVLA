"""DQNet training loop.

Optimised for a single 8GB consumer GPU (RTX 4060):
* DINOv2 features are pre-extracted (no vision backbone in memory).
* Qwen2-0.5B is wrapped with LoRA, trained in bf16, with grad checkpointing.
* Effective batch size = micro_batch * grad_accum.

Run:
    python -m GlanceVLA.dqnet.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data.dataset import LiberoChunkDataset, collate_fn, find_shards
from .models.dqnet import DQNet, DQNetConfig, trainable_parameters


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="key=value overrides, e.g. train.batch_size=4")
    parser.add_argument("--resume", default=None,
                        help="checkpoint dir to resume from (restores weights, "
                             "optimizer, scheduler, and step)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        # try parse as number/bool
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        cur[parts[-1]] = v

    torch.manual_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["train"]["precision"] == "bf16" else torch.float32

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    shards = find_shards(cfg["data"]["feature_cache_dir"])
    if not shards:
        raise RuntimeError(
            f"No feature shards in {cfg['data']['feature_cache_dir']} -- run extract_features first"
        )
    print(f"[data] {len(shards)} shards")
    dataset = LiberoChunkDataset(
        shards,
        chunk_size=cfg["data"]["chunk_size"],
        stride=cfg["data"]["stride"],
    )
    print(f"[data] {len(dataset)} chunked samples")
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )

    # --- model ---
    mcfg = DQNetConfig(
        llm_name=cfg["model"]["llm_name"],
        vision_dim=cfg["model"]["vision_dim"],
        proprio_dim=cfg["model"]["proprio_dim"],
        action_dim=cfg["model"]["action_dim"],
        chunk_size=cfg["data"]["chunk_size"],
        lora_r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"],
        gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
        use_target_token=cfg["model"].get("use_target_token", False),
        grid_size=cfg["data"].get("grid_size", 8),
        target_attn_dim=cfg["model"].get("target_attn_dim", 256),
        instruction_embed_dim=cfg["model"].get("instruction_embed_dim", 0),
    )
    model = DQNet(mcfg).to(device).to(dtype)
    print(f"[model] trainable params: {trainable_parameters(model):,}")

    # --- optimiser ---
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(
        optim, cfg["train"]["warmup_steps"], cfg["train"]["max_steps"]
    )

    grad_accum = cfg["train"]["grad_accum_steps"]
    log_every = cfg["train"]["log_every"]
    save_every = cfg["train"]["save_every"]
    max_steps = cfg["train"]["max_steps"]
    lambda_v = cfg["loss"]["lambda_vision"]
    lambda_a = cfg["loss"]["lambda_action"]
    lambda_g = cfg["loss"]["lambda_gripper"]
    lambda_t = cfg["loss"].get("lambda_target", 0.0)

    # --- resume (restore weights + optimizer + scheduler + step) ---
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        sd = torch.load(resume_dir / "trainable.pt", map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[resume] WARNING unexpected keys: {unexpected[:5]}")
        ts_path = resume_dir / "train_state.pt"
        if ts_path.exists():
            ts = torch.load(ts_path, map_location=device)
            optim.load_state_dict(ts["optimizer"])
            scheduler.load_state_dict(ts["scheduler"])
            start_step = int(ts["step"])
            print(f"[resume] restored optimizer+scheduler from step {start_step}")
        else:
            print(f"[resume] WARNING no train_state.pt in {resume_dir}; "
                  f"weights loaded but optimizer/step reset to 0")

    # --- train loop ---
    model.train()
    step = start_step
    micro_step = 0
    accum_loss = {"total": 0.0, "vision": 0.0, "action": 0.0, "gripper": 0.0, "target": 0.0}
    t0 = time.time()

    while step < max_steps:
        for batch in loader:
            out = model(batch)
            loss = (
                lambda_v * out["vision_loss"]
                + lambda_a * out["action_loss"]
                + lambda_g * out["gripper_loss"]
                + lambda_t * out["target_loss"]
            )
            (loss / grad_accum).backward()

            accum_loss["total"] += float(loss.detach())
            accum_loss["vision"] += float(out["vision_loss"].detach())
            accum_loss["action"] += float(out["action_loss"].detach())
            accum_loss["gripper"] += float(out["gripper_loss"].detach())
            accum_loss["target"] += float(out["target_loss"].detach())
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    n = grad_accum * log_every
                    elapsed = time.time() - t0
                    print(
                        f"[step {step:>6}] "
                        f"loss={accum_loss['total']/n:.4f}  "
                        f"vis={accum_loss['vision']/n:.4f}  "
                        f"act={accum_loss['action']/n:.4f}  "
                        f"grip={accum_loss['gripper']/n:.4f}  "
                        f"tgt={accum_loss['target']/n:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  "
                        f"{elapsed:.1f}s  "
                        f"mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
                    )
                    accum_loss = {k: 0.0 for k in accum_loss}
                    t0 = time.time()

                if step % save_every == 0:
                    save_checkpoint(model, output_dir / f"ckpt_{step:06d}", mcfg,
                                    optim=optim, scheduler=scheduler, step=step)

                if step >= max_steps:
                    break

    save_checkpoint(model, output_dir / "ckpt_final", mcfg,
                    optim=optim, scheduler=scheduler, step=step)
    print("[done] training finished")


def save_checkpoint(model: DQNet, path: Path, mcfg: DQNetConfig,
                    optim=None, scheduler=None, step: int = 0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # Save LoRA adapter + heads + projections (everything trainable)
    state = {k: v for k, v in model.state_dict().items()
             if any(t in k for t in [
                 "lora_", "vision_proj", "proprio_embed",
                 "action_embed", "vision_head", "action_head",
                 "target_pool", "instruction_to_query"
             ])}
    torch.save(state, path / "trainable.pt")
    # Save config
    with open(path / "config.yaml", "w") as f:
        yaml.safe_dump(vars(mcfg), f)
    # Save training state for resume (optimizer momentum, LR schedule pos, step)
    if optim is not None and scheduler is not None:
        torch.save(
            {
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            path / "train_state.pt",
        )
    print(f"[ckpt] saved to {path}")


if __name__ == "__main__":
    main()
