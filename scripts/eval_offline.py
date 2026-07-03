"""Offline evaluation of a trained DQNet checkpoint in the LIBERO simulator.

Rolls the policy out closed-loop on one or more LIBERO tasks, records the
agentview render to an mp4, and reports success / steps.

Observation -> model-input mapping (must match dqnet.data.extract_features):
    agent visual  : obs["agentview_image"]            (flip vertical -> DINOv2 CLS)
    wrist visual  : obs["robot0_eye_in_hand_image"]    (flip vertical -> DINOv2 CLS)
    proprio (8)   : [eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]

The policy predicts an action chunk of length K; we execute all K actions
open-loop, then re-observe and predict again (receding-horizon control).

Run (inside the lerobot WSL env, from repo root):
    PYTHONPATH=$PWD MUJOCO_GL=glx \
        python scripts/eval_offline.py \
        --ckpt output/run1/ckpt_final \
        --task_suite libero_object --task_id 0 \
        --n_episodes 3 --max_steps 300 --video_dir output/eval_videos
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, ".")

import robosuite.utils.transform_utils as T
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.video_utils import VideoWriter

from dqnet.models.dqnet import DQNet, DQNetConfig
from scripts.infer import load_policy, FeatureEncoder


def build_proprio(obs: dict) -> np.ndarray:
    """Reconstruct the 8-dim proprio vector used at training time."""
    ee_pos = obs["robot0_eef_pos"].astype(np.float32)            # (3,)
    ee_ori = T.quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32)  # (3,)
    grip = obs["robot0_gripper_qpos"].astype(np.float32)         # (2,)
    return np.concatenate([ee_pos, ee_ori, grip], axis=-1)       # (8,)


def _floor_id_by_largest(seg: np.ndarray) -> int:
    """Pick the floor/background instance id as the largest-area segment.
    Must match extract_features_masked.py (largest area is robust on both cams;
    corners-vote failed on the wrist cam where the arm fills the top corners)."""
    ids, counts = np.unique(seg, return_counts=True)
    return int(ids[np.argmax(counts)])


def _blur_floor(rgb: np.ndarray, seg: np.ndarray, radius: float) -> np.ndarray:
    """Blur the floor region of rgb. Must match extract_features_masked.py."""
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius)))
    mask = seg == _floor_id_by_largest(seg)
    out = rgb.copy()
    out[mask] = blurred[mask]
    return out


def _seg_for(obs: dict, cam: str) -> np.ndarray:
    return np.asarray(next(obs[k] for k in obs if cam in k and "seg" in k.lower())).squeeze()


def _instruction_embed(text: str, model_name: str, device: torch.device) -> torch.Tensor | None:
    """Deprecated: instruction conditioning is now internal to the model
    (query = LLM last-layer hidden state). Kept as a no-op stub."""
    return None


@torch.no_grad()
def predict_chunk(
    model: DQNet,
    encoder: FeatureEncoder,
    obs: dict,
    instruction: str,
    device: torch.device,
    mask_floor: bool = False,
    blur_radius: float = 12.0,
) -> np.ndarray:
    """Encode one observation and return the predicted action chunk (K, 7)."""
    agent_img = obs["agentview_image"]                    # (H, W, 3) uint8
    wrist_img = obs["robot0_eye_in_hand_image"]
    if mask_floor:
        agent_img = _blur_floor(agent_img, _seg_for(obs, "agentview"), blur_radius)
        wrist_img = _blur_floor(wrist_img, _seg_for(obs, "eye_in_hand"), blur_radius)
    agent_feat = encoder(agent_img[None]).to(device)      # (1, 768)
    wrist_feat = encoder(wrist_img[None]).to(device)
    proprio = torch.from_numpy(build_proprio(obs)[None]).to(device)  # (1, 8)

    # Supervised-target-token models need the agentview patch grid. The target
    # token is computed by the attention pool (query = LLM instruction hidden
    # state, extracted internally) -- no segmentation/GT needed here.
    cur_agent_patch = None
    if getattr(model, "target_pool", None) is not None:
        cur_agent_patch = encoder.patches(
            agent_img[None], grid_size=model.cfg.grid_size
        ).to(device)                                      # (1, G*G, 768)

    actions = model.predict_action_chunk(
        instructions=[instruction],
        cur_agent=agent_feat,
        cur_wrist=wrist_feat,
        cur_proprio=proprio,
        cur_agent_patch=cur_agent_patch,
    )
    return actions[0].float().cpu().numpy()               # (K, 7)


def run_episode(
    env,
    model,
    encoder,
    instruction,
    init_state,
    max_steps,
    device,
    video_writer,
    settle_steps=5,
    exec_horizon=0,
    mask_floor=False,
) -> tuple[bool, int]:
    env.reset()
    obs = env.set_init_state(init_state)
    # optional no-op steps to let physics settle (LIBERO benchmark convention)
    for _ in range(max(0, settle_steps)):
        obs, _, _, _ = env.step([0.0] * 7)

    K = model.cfg.chunk_size
    # How many of the K predicted actions to execute open-loop before
    # re-observing. <=0 or >=K means execute the whole chunk (original behavior).
    h = K if exec_horizon <= 0 else min(exec_horizon, K)
    success = False
    steps = 0
    while steps < max_steps:
        chunk = predict_chunk(model, encoder, obs, instruction, device, mask_floor=mask_floor)
        for k in range(h):
            obs, reward, done, info = env.step(chunk[k].tolist())
            video_writer.append_vector_obs([obs], [done], camera_name="agentview_image")
            steps += 1
            if done:
                success = True
                break
        if success:
            break
    return success, steps


def setup_from_suite(args):
    """Benchmark mode: env + init states from a task suite (uses task.language,
    which for LIBERO-plus includes perturbation suffixes like 'table 1')."""
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    instruction = task.language
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(args.task_id)
    print(f"[eval] mode=suite suite={args.task_suite} task_id={args.task_id}")
    return bddl, instruction, init_states, 5


def setup_from_demo(args):
    """Ground-truth scene mode: env + init state from a recorded demo's OWN bddl
    (base task, clean instruction, default floor). Matches replay_gt.py exactly,
    so the model is evaluated on its in-distribution scene."""
    import json
    import h5py

    with h5py.File(args.hdf5, "r") as f:
        data_attrs = dict(f["data"].attrs)
        instruction = json.loads(data_attrs["problem_info"])["language_instruction"]
        # Collect one init state (states[0]) from each of N demos, so a single
        # run can sweep many initial conditions without reloading the model.
        all_demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.n_init and args.n_init > 1:
            chosen = all_demos[:args.n_init]
        else:
            chosen = [args.demo]
        init_states = [f[f"data/{d}/states"][:][0] for d in chosen]

    raw = data_attrs["bddl_file_name"]
    bddl = os.path.join(get_libero_path("bddl_files"), Path(raw).parent.name, Path(raw).name)
    # Optional: override the bddl (e.g. a floor/table appearance variant) while
    # keeping the demo's recorded init state, so the ONLY change is the background.
    if args.bddl:
        bddl = args.bddl if os.path.isabs(args.bddl) else os.path.join(
            get_libero_path("bddl_files"), Path(raw).parent.name, args.bddl)
        print(f"[eval] bddl override -> {Path(bddl).name}")
    print(f"[eval] mode=demo hdf5={Path(args.hdf5).name} n_init={len(init_states)}")
    # Use the demo's exact recorded initial sim state; no settle steps (match recording).
    return bddl, instruction, init_states, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="output/run1/ckpt_final")
    parser.add_argument("--task_suite", default="libero_object")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--video_dir", default="output/eval_videos")
    parser.add_argument("--dinov2_name", default="facebook/dinov2-base")
    parser.add_argument("--camera_size", type=int, default=128)
    parser.add_argument("--hdf5", default=None,
                        help="If set, evaluate on this demo's OWN base scene "
                             "(ground-truth bddl + recorded init state + clean instruction).")
    parser.add_argument("--demo", default="demo_0",
                        help="Demo key inside --hdf5 to take the init state from.")
    parser.add_argument("--n_init", type=int, default=0,
                        help="In --hdf5 mode, sweep init states from the first N demos "
                             "(demo_0..demo_{N-1}) in one run. 0/1 = single demo.")
    parser.add_argument("--bddl", default=None,
                        help="Override bddl in --hdf5 mode (filename under the same "
                             "problem folder, or absolute path). Use a floor/table "
                             "variant to change only the background.")
    parser.add_argument("--exec_horizon", type=int, default=0,
                        help="Execute only the first N of K predicted actions before "
                             "re-observing (receding horizon). 0 = execute full chunk.")
    parser.add_argument("--mask_floor", action="store_true",
                        help="Blur the floor region before encoding (must match how the "
                             "checkpoint was trained, i.e. extract_features_masked.py).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    model = load_policy(args.ckpt, device, dtype)
    encoder = FeatureEncoder(model_name=args.dinov2_name, device=device)

    # --- task / env ---
    if args.hdf5:
        bddl, instruction, init_states, settle = setup_from_demo(args)
    else:
        bddl, instruction, init_states, settle = setup_from_suite(args)
    print(f"[eval] instruction: {instruction!r}")
    print(f"[eval] bddl: {bddl}")
    print(f"[eval] mask_floor: {args.mask_floor}")

    env_kwargs = dict(
        bddl_file_name=bddl,
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    if args.mask_floor:
        # need instance segmentation in obs to locate the floor at inference time
        env_kwargs["camera_segmentations"] = "instance"
    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(0)

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_writer = VideoWriter(str(video_dir), save_video=True)

    n_success = 0
    # In n_init sweep mode, evaluate every collected init state; otherwise cap by n_episodes.
    n = len(init_states) if (args.hdf5 and args.n_init and args.n_init > 1) else min(args.n_episodes, len(init_states))
    steps_list = []
    for ep in range(n):
        success, steps = run_episode(
            env, model, encoder, instruction,
            init_states[ep], args.max_steps, device, video_writer, settle,
            exec_horizon=args.exec_horizon, mask_floor=args.mask_floor,
        )
        n_success += int(success)
        steps_list.append((ep, success, steps))
        print(f"[eval] episode {ep}: success={success} steps={steps}", flush=True)

    video_writer.save()
    env.close()
    print(f"[eval] SUCCESS RATE: {n_success}/{n} = {n_success / n:.1%}")
    print(f"[eval] video saved under {video_dir}")


if __name__ == "__main__":
    main()
