"""Extract DINOv2 features + a SUPERVISED TARGET HEATMAP for each frame.

Method 3 (supervised target token)
----------------------------------
For every demo frame we additionally store:
  * agent_patch          (T, G*G, 768) fp16 -- DINOv2 patch grid, 16x16 avg-pooled
                          down to GxG, aligned with the encoder's flipped image.
  * agent_target_heatmap (T, G*G)      fp32 -- ground-truth attention target:
                          the obj_of_interest instance mask, downsampled to GxG
                          and normalized to sum 1. Supervises TargetAttentionPool.
  * target_valid         (T,)          bool -- False when the target is not visible
                          in agentview (empty mask); those frames are masked out
                          of the attention loss.

The target mask comes from instance segmentation via SegmentationRenderEnv
(`env.instance_to_id[obj]` for obj in `env.obj_of_interest`). NO 3D camera
projection is needed. Segmentation is only used HERE (offline); at inference
the target token is produced by the learned attention pool with no GT.

We reproduce the recorded trajectory by restoring each timestep's sim state
(states[t]), so RGB / patches / heatmap / actions / proprio stay aligned with
the original demo, exactly like extract_features_masked.py.

Output matches the keys LiberoChunkDataset expects (agent_feat, wrist_feat,
actions, proprio) PLUS the three target keys above. Written to a SEPARATE cache
dir so it never clobbers the CLS-only features.

Run (lerobot WSL env, repo root):
    PYTHONPATH=$PWD:$PWD MUJOCO_GL=glx HF_HUB_OFFLINE=1 \
      python scripts/extract_features_target.py \
        --hdf5 datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
        --out_dir cache/dino_features_target --grid_size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, ".")

from libero.libero import get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from scripts.infer import FeatureEncoder


def seg_key_for(obs: dict, cam: str) -> str:
    return next(k for k in obs if cam in k and "seg" in k.lower())


def instruction_embed_from_text(text: str, tokenizer, model, device, max_tokens: int = 32) -> np.ndarray:
    """Extract CLS-like embedding from instruction text using a pre-trained LLM.

    Uses the [BOS] token embedding as a proxy for the instruction representation.
    This is cheaper than a full forward pass and works well as a conditioning signal.
    """
    tokens = tokenizer(text, truncation=True, max_length=max_tokens,
                       return_tensors="pt")
    ids = tokens.input_ids.to(device)
    # Use the [BOS]/[PAD] token embedding directly (no forward pass needed).
    # This captures lexical information efficiently.
    with torch.no_grad():
        # Get embedding from the input embedding layer
        embed_layer = model.get_input_embeddings()
        emb = embed_layer(ids).mean(dim=1)  # (1, D)
    return emb.float().cpu().numpy().squeeze()  # (D,)


def heatmap_from_mask(mask: np.ndarray, grid_size: int, flip_vertical: bool) -> np.ndarray:
    """Binary (H, W) target mask -> (grid_size*grid_size,) distribution summing to 1.

    The encoder flips images vertically before extracting patches, so we flip
    the mask the same way to keep the heatmap aligned with the patch grid.
    Returns a zero vector if the mask is empty (caller marks frame invalid).
    """
    m = mask.astype(np.float32)
    if flip_vertical:
        m = m[::-1].copy()
    t = torch.from_numpy(m)[None, None]                       # (1,1,H,W)
    pooled = F.adaptive_avg_pool2d(t, output_size=(grid_size, grid_size))
    vec = pooled.reshape(-1).numpy().astype(np.float32)       # (G*G,)
    s = vec.sum()
    if s <= 1e-8:
        return np.zeros_like(vec)
    return vec / s


def target_mask(env, obs: dict) -> np.ndarray:
    """Union of obj_of_interest instance masks in agentview, as a bool (H, W)."""
    seg = np.asarray(obs[seg_key_for(obs, "agentview")]).squeeze()
    out = np.zeros_like(seg, dtype=bool)
    for obj in env.obj_of_interest:
        sid = env.instance_to_id.get(obj)
        if sid is not None:
            out |= (seg == sid)
    return out


def process_demo(env, encoder, states, grid_size, batch_size):
    """Re-render every frame; return agent CLS, agent patches, wrist CLS,
    target heatmaps, and validity flags -- all aligned with `states`."""
    agent_imgs, wrist_imgs, heatmaps, valids = [], [], [], []
    for t in range(len(states)):
        obs = env.set_init_state(states[t]) if t == 0 else env.regenerate_obs_from_state(states[t])
        agent_imgs.append(obs["agentview_image"])
        wrist_imgs.append(obs["robot0_eye_in_hand_image"])
        mask = target_mask(env, obs)
        hm = heatmap_from_mask(mask, grid_size, encoder.flip_vertical)
        heatmaps.append(hm)
        valids.append(bool(mask.any()))

    def encode_cls(imgs):
        feats = []
        for i in range(0, len(imgs), batch_size):
            chunk = np.stack(imgs[i:i + batch_size], 0)
            feats.append(encoder(chunk).cpu().numpy().astype(np.float16))
        return np.concatenate(feats, 0)

    def encode_patches(imgs):
        feats = []
        for i in range(0, len(imgs), batch_size):
            chunk = np.stack(imgs[i:i + batch_size], 0)
            feats.append(encoder.patches(chunk, grid_size=grid_size).cpu().numpy().astype(np.float16))
        return np.concatenate(feats, 0)

    agent_feat = encode_cls(agent_imgs)
    agent_patch = encode_patches(agent_imgs)
    wrist_feat = encode_cls(wrist_imgs)
    heatmaps = np.stack(heatmaps, 0).astype(np.float32)
    valids = np.array(valids, dtype=bool)
    return agent_feat, agent_patch, wrist_feat, heatmaps, valids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5")
    ap.add_argument("--out_dir", default="cache/dino_features_target")
    ap.add_argument("--dinov2_name", default="facebook/dinov2-base")
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_demos", type=int, default=None, help="limit demos (debug)")
    ap.add_argument("--llm_name", default="cache/models/Qwen/Qwen2-0___5B",
                    help="LLM for instruction embedding (CLS token)")
    ap.add_argument("--embed_instruction", action="store_true",
                    help="Extract instruction CLS embedding and store in NPZ")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FeatureEncoder(model_name=args.dinov2_name, device=device)

    # Precompute instruction embedding if requested
    instruction_embed = None
    if args.embed_instruction:
        from transformers import AutoTokenizer, AutoModel
        print(f"[info] loading instruction encoder {args.llm_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        model = AutoModel.from_pretrained(args.llm_name).to(device, dtype=torch.float16).eval()

    with h5py.File(args.hdf5, "r") as f:
        data_attrs = dict(f["data"].attrs)
        instruction = json.loads(data_attrs["problem_info"])["language_instruction"]

        # Compute instruction embedding now that we have the text
        if args.embed_instruction:
            instruction_embed = instruction_embed_from_text(instruction, tokenizer, model, device)
            print(f"[info] instruction embed dim={instruction_embed.shape}")

        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.max_demos:
            demo_keys = demo_keys[:args.max_demos]
        demos = {}
        for d in demo_keys:
            g = f[f"data/{d}"]
            ee_pos = g["obs/ee_pos"][:].astype(np.float32)
            ee_ori = g["obs/ee_ori"][:].astype(np.float32)
            grip = g["obs/gripper_states"][:].astype(np.float32)
            demos[d] = {
                "states": g["states"][:],
                "actions": g["actions"][:].astype(np.float32),
                "proprio": np.concatenate([ee_pos, ee_ori, grip], -1),
            }

    raw = data_attrs["bddl_file_name"]
    bddl = str(Path(get_libero_path("bddl_files")) / Path(raw).parent.name / Path(raw).name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.hdf5).stem

    env = SegmentationRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128,
        camera_segmentations="instance",
    )
    env.seed(0)
    env.reset()  # builds env.instance_to_id mapping
    print(f"[target] obj_of_interest={env.obj_of_interest} "
          f"ids={[env.instance_to_id.get(o) for o in env.obj_of_interest]}", flush=True)

    total = len(demo_keys)
    for di, d in enumerate(demo_keys):
        out_path = out_dir / f"{stem}__{d}.npz"
        if out_path.exists():
            print(f"[{di+1}/{total}] {d}: already done, skip", flush=True)
            continue

        a_feat, a_patch, w_feat, heat, valid = process_demo(
            env, encoder, demos[d]["states"], args.grid_size, args.batch_size
        )
        payload = {
            f"{d}/agent_feat": a_feat,
            f"{d}/agent_patch": a_patch,
            f"{d}/wrist_feat": w_feat,
            f"{d}/agent_target_heatmap": heat,
            f"{d}/target_valid": valid,
            f"{d}/actions": demos[d]["actions"],
            f"{d}/proprio": demos[d]["proprio"],
        }
        payload["instruction"] = np.array(instruction)
        payload["demo_keys"] = np.array(sorted(set(k.split("/")[0] for k in payload if "/" in k)))
        if instruction_embed is not None:
            payload["instruction_embed"] = instruction_embed.astype(np.float32)
        # atomic write: tmp then rename
        tmp = out_dir / f"{stem}__{d}.tmp.npz"
        np.savez_compressed(tmp, **payload)
        tmp.replace(out_path)
        vis = int(valid.sum())
        print(f"[{di+1}/{total}] {d}: T={len(valid)} target_visible={vis} -> {out_path.name}", flush=True)

    env.close()
    print(f"[done] {total} demos -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
