# GlanceVLA — Vision-Language-Action Model with Supervised Target Attention

A lightweight VLA policy network for language-guided robotic manipulation, featuring a **supervised target token** that learns to focus on the manipulation object without any ground-truth segmentation at inference time.

[📄 Paper] | [🤗 Model] | [🌐 Website]

## Overview

GlanceVLA extends the DQNet(DINOv2 + Qwen2) architecture with a single **learned-query cross-attention module** over the DINOv2 patch grid. This module produces one extra token per observation frame that attends to the target object, supervised during training by the object's segmentation mask. At inference time, the token is generated entirely from the learned attention — no ground-truth masks, no 3D camera projection, no segmentation needed.

**Key properties:**
- **Tiny footprint:** ~15.7M trainable parameters (Qwen2-0.5B frozen, LoRA r=16). Runs on a single RTX 4060 (8GB).
- **offline end-to-end vision:** DINOv2 features are pre-extracted. No vision transformer in memory during training.
- **Action chunking:** Predicts K=6 future actions per inference, executed with receding-horizon control.
- **Target token:** One learned query over a pooled 8×8 patch grid (~1.5M params), supervised toward the `obj_of_interest` heatmap. Conditioned on the task instruction for semantic grounding.

## Architecture

```
Input sequence per sample:
  [language tokens] [V_agent_t] [V_target_t] [V_wrist_t] [V_proprio_t] [BOS]
  [V_agent_{t+1}] [V_wrist_{t+1}] [ACT_1]
  [V_agent_{t+2}] [V_wrist_{t+2}] [ACT_2]
  ...
  [V_agent_{t+K}] [V_wrist_{t+K}] [ACT_K] [EOS]
```

Only the current frame gains a 4th token (`V_target_t`), produced by `TargetAttentionPool` — a 1-query cross-attention over G×G pooled DINOv2 patches. Future-frame triplets and all action-decoding positions are unchanged.

**Modules:**
- **VisionProj** — DINOv2 CLS (768-d) → LLM embedding (896-d)
- **ProprioEmbed** — EE position/orientation/gripper (8-d) → LLM embedding
- **ActionEmbed** — Action (7-d) → LLM embedding (teacher-forcing)
- **TargetAttentionPool** — 1 learnable query over G×G patch grid → target token + attention logits
- **VisionHead** — LLM hidden → DINOv2 CLS (for future-frame prediction)
- **ActionHead** — LLM hidden → 7-d action

**Loss:**
```
total = λ_v·vision + λ_a·action + λ_g·gripper + λ_t·target
```
- `vision`: cosine similarity loss on future agent+wrist features
- `action`: L1 loss on delta pose (6-d)
- `gripper`: BCE on gripper state {-1,+1} → {0,1}
- `target`: cross-entropy between attention softmax and GT heatmap (masked by visibility)

## Quick Start

### Setup

```bash
git clone <repo-url>
cd GlanceVLA

# Dependencies
pip install torch transformers peft h5py numpy pillow
pip install robosuite bddl gymnasium libero-libero

# Download Qwen2-0.5B
mkdir -p cache/models/Qwen/Qwen2-0___5B
huggingface-cli download Qwen/Qwen2-0.5B --local-dir cache/models/Qwen/Qwen2-0___5B

# Download DINOv2
huggingface-cli download facebook/dinov2-base --local-dir cache/models/dinov2-base
```

### Step 1: Extract Features

```bash
# Supervised target token features (includes patch grid + GT heatmap)
PYTHONPATH=$PWD MUJOCO_GL=glx python scripts/extract_features_target.py \
  --hdf5 datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --out_dir cache/dino_features_target \
  --grid_size 8
```

### Step 2: Train

```bash
# Supervised target token (main version)
PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python -m dqnet.train --config configs/target.yaml

# Resume from checkpoint
PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python -m dqnet.train \
  --config configs/target.yaml \
  --resume output/run_target/ckpt_010000
```

### Step 3: Evaluate

```bash
# Closed-loop evaluation with video
PYTHONPATH=$PWD MUJOCO_GL=glx HF_HUB_OFFLINE=1 python scripts/eval_offline.py \
  --ckpt output/run_target/ckpt_020000 \
  --hdf5 datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --n_init 10 --max_steps 300 --exec_horizon 1 \
  --video_dir output/eval_varied_init
```

### Step 4: Visualize Target Attention

```bash
# Overlay GT heatmap vs learned attention on RGB frames
PYTHONPATH=$PWD MUJOCO_GL=glx HF_HUB_OFFLINE=1 python scripts/viz_target_attention.py \
  --ckpt output/run_target/ckpt_020000 \
  --demo demo_0 --frames 0 0.33 0.66 0.99 \
  --out output/viz_target/attention.png
```

## Experimental Results

### Supervised Target Token Quality

After 20,000 training steps on a single task:

| Metric | Value |
|--------|-------|
| argmax-hit (top cell in GT support) | **1.00** |
| top1-in-top5 | **1.00** |
| cosine(GT, attn) | **0.987** |
| KL(GT ‖ attn) | 0.037 |

The target token learns to look at the manipulation object within a few thousand steps.

### Closed-Loop Success Rate

| Condition | Success Rate |
|-----------|-------------|
| Same scene, 10 different initial states (demo_0~9) | **80%** (8/10) |
| Same scene, 10 different initial states (demo_40~49) | **80%** (8/10) |
| Unseen floor texture (tb_3) | **50%** (5/10) |
| Different task (bbq sauce) | **0%** (0/10) |

> **Finding:** The target token's attention is robust to background changes (OOD argmax-hit = 100%), but the overall policy fails because the global CLS token depends on background. The bottleneck is in the CLS representation, not the target token.

### Training Loss Progression

| Step | Total | Vision | Action | Gripper | Target |
|------|-------|--------|--------|---------|--------|
| 2,500 | 2.33 | 0.24 | 0.04 | 0.06 | 3.97 |
| 10,000 | 0.97 | 0.02 | 0.00 | 0.00 | 1.89 |
| 20,000 | 0.96 | 0.02 | 0.00 | 0.00 | 1.89 |

> **Note:** Total loss stabilizes around 0.96 because the target term (cross-entropy between attention and soft GT heatmap) has an entropy lower bound (~1.73). Monitor `action_loss` and `vision_loss` for training progress; total loss is dominated by the target term and insensitive to policy improvements.

### Training Efficiency

| Metric | Value |
|--------|-------|
| Trainable parameters | 15.7M |
| GPU memory | ~1.2 GB |
| Training speed | ~35 steps/min (batch_size=8) |
| Inference time | ~3 min/episode |

## Ablation Study Design

Three configurations to isolate the target token contribution:

| Variant | Config | Purpose |
|---------|--------|---------|
| Baseline | `use_target_token=False` | CLS-only reference |
| Unsupervised target token | `lambda_target=0` | Test attention mechanism alone |
| Supervised target token | `lambda_target=0.5` | Full method |

Evaluated on LIBERO success rate, LIBERO-plus perturbation dimensions, and attention-GT alignment.

## Known Issues & Engineering Notes

- **exec_horizon=1 is required.** Full-chunk autoregressive rollout (6 steps) causes drift accumulation that makes the policy fail. Always use `--exec_horizon 1` in evaluation/deployment.
- **Loss floor:** Total loss stabilizes around 0.96 because the target term (cross-entropy between attention and soft GT heatmap) has an entropy lower bound (~1.73). Monitor `action_loss` and `vision_loss` for training progress; total loss is dominated by the target term and insensitive to policy improvements.
- **Target token uses union of obj_of_interest.** By default, the supervision signal combines all objects listed in `obj_of_interest` (e.g., both the soup and the basket). To focus on a single manipuland, edit `target_mask()` in `extract_features_target.py`.
- **Instruction-aware target token.** The target token's query is conditioned on the task instruction for semantic grounding.

## Citation

If you use this code for your research, please cite:

```bibtex
@article{gazevla2025,
  title={GlanceVLA: Target-Focused Supervision for Lightweight Vision-Language-Action Models},
  author={},
  journal={arXiv preprint arXiv:{xxxx.xxxxx}},
  year={2025},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
