# GlanceVLA — 基于监督式目标注意力的轻量级视觉-语言-动作模型

一个面向语言引导机器人操作的轻量级 VLA 策略网络，核心创新是引入**监督式目标 token**——在训练时通过目标物体的分割掩码监督注意力，推理时完全无需任何 ground truth。

[📄 论文] | [🤗 模型] | [🌐 网站]

## 概述

GlanceVLA 在 DQNet(DINOv2 + Qwen2) 架构基础上增加了一个**可学习查询的交叉注意力模块**，对 DINOv2 的 patch 网格做注意力池化。该模块在每个观测帧产生一个额外的 token，专注于目标物体，训练时通过目标物体的分割掩码进行监督。推理时该 token 完全由学到的注意力生成——不需要分割掩码、不需要 3D 相机投影、不需要任何 ground truth。

**核心特性：**
- **极小的参数量：** 可训练参数仅 ~15.7M（Qwen2-0.5B 冻结，LoRA r=16）。单张 RTX 4060（8GB）即可运行。
- **离线端到端编码视觉：** DINOv2 特征离线预提取，训练时无需视觉 Transformer 驻留内存。
- **动作分块预测：** 每次推理预测 K=6 步未来动作，采用滚动式控制执行。
- **监督式目标 token：** 1 个可学习Query对 8×8 池化 patch 网格做交叉注意力（~1.5M 参数量），训练时监督 `obj_of_interest` heatmap。推理时完全自主，无需分割。

## 架构

```
样本输入序列：
  [语言 tokens] [V_agent_t] [V_target_t] [V_wrist_t] [V_proprio_t] [BOS]
  [V_agent_{t+1}] [V_wrist_{t+1}] [ACT_1]
  [V_agent_{t+2}] [V_wrist_{t+2}] [ACT_2]
  ...
  [V_agent_{t+K}] [V_wrist_{t+K}] [ACT_K] [EOS]
```

仅当前帧增加第 4 个 token（`V_target_t`），由 `TargetAttentionPool` 产生——1 个可学习Query对 G×G 池化 DINOv2 patch 做交叉注意力。未来帧三元组和所有动作解码位置不变。

**模块：**
- **VisionProj** — DINOv2 CLS (768-d) → LLM 嵌入 (896-d)
- **ProprioEmbed** — 末端位置/姿态/夹爪 (8-d) → LLM 嵌入
- **ActionEmbed** — 动作 (7-d) → LLM 嵌入（教师强制）
- **TargetAttentionPool** — 1 个可学习查询对 G×G patch 网格 → 目标 token + 注意力 logits
- **VisionHead** — LLM 隐藏态 → DINOv2 CLS（用于未来帧预测）
- **ActionHead** — LLM 隐藏态 → 7-d 动作

**损失：**
```
total = λ_v·vision + λ_a·action + λ_g·gripper + λ_t·target
```
- `vision`：未来 agent+wrist 特征的余弦相似度损失
- `action`：delta 位姿 (6-d) 的 L1 损失
- `gripper`：夹爪状态 {-1,+1} → {0,1} 的 BCE
- `target`：注意力 softmax 与 GT heatmap 的交叉熵（按可见性掩码）

## 快速开始

### 环境搭建

```bash
git clone <repo-url>
cd GlanceVLA

# 依赖
pip install torch transformers peft h5py numpy pillow
pip install robosuite bddl gymnasium libero-libero

# 下载 Qwen2-0.5B
mkdir -p cache/models/Qwen/Qwen2-0___5B
huggingface-cli download Qwen/Qwen2-0.5B --local-dir cache/models/Qwen/Qwen2-0___5B

# 下载 DINOv2
huggingface-cli download facebook/dinov2-base --local-dir cache/models/dinov2-base
```

### 第一步：提取特征

```bash
# 监督式目标 token 特征（含 patch 网格 + GT heatmap）
PYTHONPATH=$PWD MUJOCO_GL=glx python scripts/extract_features_target.py \
  --hdf5 datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --out_dir cache/dino_features_target \
  --grid_size 8
```

### 第二步：训练

```bash
# 监督式目标 token（主版本）
PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python -m dqnet.train --config configs/target.yaml

# 从 checkpoint 恢复训练
PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python -m dqnet.train \
  --config configs/target.yaml \
  --resume output/run_target/ckpt_010000
```

### 第三步：评测

```bash
# 闭环评测 + 视频
PYTHONPATH=$PWD MUJOCO_GL=glx HF_HUB_OFFLINE=1 python scripts/eval_offline.py \
  --ckpt output/run_target/ckpt_020000 \
  --hdf5 datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --n_init 10 --max_steps 300 --exec_horizon 1 \
  --video_dir output/eval_varied_init
```

### 第四步：可视化目标注意力

```bash
# 在 RGB 帧上叠加 GT heatmap 与学到的注意力
PYTHONPATH=$PWD MUJOCO_GL=glx HF_HUB_OFFLINE=1 python scripts/viz_target_attention.py \
  --ckpt output/run_target/ckpt_020000 \
  --demo demo_0 --frames 0 0.33 0.66 0.99 \
  --out output/viz_target/attention.png
```

## 实验结果

### 监督式目标 token 质量

单任务训练 20,000 步后：

| 指标 | 值 |
|------|-----|
| argmax-hit（top 单元格在 GT 支撑域内） | **1.00** |
| top1-in-top5 | **1.00** |
| cosine(GT, attn) | **0.987** |
| KL(GT ‖ attn) | 0.037 |

目标 token 在几千步内就学会了看向操作目标。

### 闭环成功率

| 条件 | 成功率 |
|------|--------|
| 同场景，10 个不同初始状态（demo_0~9） | **80%** (8/10) |
| 同场景，10 个不同初始状态（demo_40~49） | **80%** (8/10) |
| 未见过的地板纹理（tb_3） | **50%** (5/10) |
| 不同任务（bbq sauce） | **0%** (0/10) |

> **关键发现：** 目标 token 的注意力对背景变化具有鲁棒性（OOD 场景下 argmax-hit = 100%），但整体策略在 OOD 上失败，因为全局 CLS token 依赖于背景。瓶颈在 CLS 表征，不在目标 token。

https://github.com/user-attachments/assets/cf32a031-0c5b-4951-b3ec-558373fcafe3
https://github.com/user-attachments/assets/25460f42-4dec-4c79-b56d-c8c6c9996bb6

### 训练损失演化

| 步数 | Total | Vision | Action | Gripper | Target |
|------|-------|--------|--------|---------|--------|
| 2,500 | 2.33 | 0.24 | 0.04 | 0.06 | 3.97 |
| 10,000 | 0.97 | 0.02 | 0.00 | 0.00 | 1.89 |
| 20,000 | 0.96 | 0.02 | 0.00 | 0.00 | 1.89 |

> **注意：** Total loss 稳定在 0.96 附近是因为 target 项（注意力与软 GT heatmap 的交叉熵）有熵下限（~1.73）。应关注 `action_loss` 和 `vision_loss` 判断训练进展；total loss 被 target 项主导，对策略改进不敏感。

### 训练效率

| 指标 | 值 |
|------|-----|
| 可训练参数 | 15.7M |
| GPU 显存 | ~1.2 GB |
| 训练速度 | ~35 步/分钟（batch_size=8） |
| 推理时间 | ~3 分钟/episode |

## 消融实验设计

三种配置隔离目标 token 的贡献：

| 变体 | 配置 | 目的 |
|------|------|------|
| 基线 | `use_target_token=False` | 仅 CLS 对照 |
| 无监督目标 token | `lambda_target=0` | 仅验证注意力机制 |
| 监督式目标 token | `lambda_target=0.5` | 完整方法 |

评测维度：LIBERO 成功率、LIBERO-plus 扰动维度、注意力-GT 对齐度。

## 已知问题与工程笔记

- **必须使用 exec_horizon=1。** 全 chunk 自回归 rollout（6 步）会导致漂移累积，策略失败。评测/部署时务必使用 `--exec_horizon 1`。
- **损失地板效应：** Total loss 稳定在 0.96 附近，因为 target 项有熵下限（~1.73）。应关注 `action_loss` 和 `vision_loss` 判断训练进展。
- **目标 token 使用 obj_of_interest 的并集。** 默认监督信号合并了 `obj_of_interest` 中的所有物体（如 soup 和 basket）。如需聚焦单一操作物体，编辑 `extract_features_target.py` 中的 `target_mask()`。
- **指令感知的目标 token。** 目标 token 的查询受任务指令条件调制，提供语义 grounding。

## 引用

```bibtex
@article{gazevla2025,
  title={GlanceVLA: Target-Focused Supervision for Lightweight Vision-Language-Action Models},
  author={},
  journal={arXiv preprint arXiv:{xxxx.xxxxx}},
  year={2025},
}
```

## 许可

MIT License. 详见 [LICENSE](LICENSE)。
