"""DQNet model.

Architecture
------------
Token sequence consumed by the LLM (per sample, batch dim omitted)::

    [language tokens (variable length)]
    [V_agent_t]  [V_wrist_t]  [V_proprio_t]                # current frame: 3 tokens
    [BOS]
    [V_agent_{t+1}] [V_wrist_{t+1}] [ACT_1]                # future frame 1: 2 vis + 1 act
    [V_agent_{t+2}] [V_wrist_{t+2}] [ACT_2]
    ...
    [V_agent_{t+K}] [V_wrist_{t+K}] [ACT_K]
    [EOS]

When `use_target_token=True`, the current frame carries a 4th token, a
supervised "target token" produced by `TargetAttentionPool` over the agentview
patch grid (method 3). Crucially, the attention query is the LLM's own last-layer
hidden state at the end of the instruction tokens — extracted in a first forward
pass. This makes the attention **conditioned on the task language** (e.g. "pick up
the soup" vs "place in the basket") rather than a generic visual pattern.

    [V_agent_t] [V_target_t] [V_wrist_t] [V_proprio_t]     # current frame: 4 tokens

Only the current frame changes; the future-frame triplets and all action-
decoding positions are unaffected (offsets derive from cur_embeds.size(1)).

All multimodal tokens enter the LLM through `inputs_embeds`. We never put
extra tokens in the tokenizer vocabulary; instead we re-use the LLM's
existing BOS/EOS embeddings and feed continuous embeddings directly for
everything else.

Heads
-----
* `vision_head`  : LLM hidden -> 768-d vector. Trained with cosine loss
                   against future DINOv2 CLS tokens at the positions where
                   we *predict* a visual token. Concretely, for the k-th
                   future frame we read hidden states at positions:
                       hidden_at(V_agent_{t+k} input) -> predicts V_agent_{t+k}
                       hidden_at(V_wrist_{t+k} input) -> predicts V_wrist_{t+k}
                   But causal attention means each position attends only
                   to past tokens. So we shift: the prediction of
                   V_agent_{t+k} is read from the position *immediately
                   before* the V_agent_{t+k} input embedding (BOS for k=1,
                   ACT_{k-1} for k>1).
* `action_head`  : MLP 7-dim. Read at positions [V_wrist_{t+k}] to predict
                   action_k.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DQNetConfig:
    llm_name: str = "cache/models/Qwen/Qwen2-0___5B"
    vision_dim: int = 768
    proprio_dim: int = 8
    action_dim: int = 7
    chunk_size: int = 6
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    max_lang_tokens: int = 32  # truncate long task strings
    # --- supervised target token (method 3) ---
    use_target_token: bool = False  # add a 4th current-frame token via TargetAttentionPool
    grid_size: int = 8              # GxG pooled patch grid the query attends over (G*G patches)
    target_attn_dim: int = 256      # internal q/k/v dim of the attention pool
    instruction_embed_dim: int = 0  # 0 = no instruction conditioning; >0 enables instruction-aware query


class ProprioEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionProj(nn.Module):
    """Project DINOv2 CLS tokens into the LLM embedding space."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEmbed(nn.Module):
    """Embed a 7-dim action into LLM embedding space (used for teacher forcing)."""

    def __init__(self, action_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TargetAttentionPool(nn.Module):
    """Instruction-conditioned cross-attention over a GxG patch grid.

    The query is the LLM's last-layer hidden state at the end of the
    instruction tokens — a semantically rich, task-aware vector. This
    replaces the old random learnable query, making the attention
    "know" what the task is (e.g. "pick up the soup" vs "place in the
    basket") rather than just memorizing a generic visual pattern.

    Input : patches (B, N, vision_dim), query (B, attn_dim)
    Output: token   (B, out_dim), logits (B, N)
    """

    def __init__(self, vision_dim: int, out_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.to_k = nn.Linear(vision_dim, attn_dim)
        self.to_v = nn.Linear(vision_dim, attn_dim)
        self.scale = attn_dim ** -0.5
        self.proj = nn.Sequential(
            nn.Linear(attn_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, patches: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patches: (B, N, vision_dim), query: (B, attn_dim)
        x = self.norm(patches)
        k = self.to_k(x)                                   # (B, N, A)
        v = self.to_v(x)                                   # (B, N, A)
        q = query.unsqueeze(1)                              # (B, 1, A)
        logits = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (B, 1, N)
        logits = logits.squeeze(1)                         # (B, N)
        weights = torch.softmax(logits, dim=-1)            # (B, N)
        pooled = torch.matmul(weights.unsqueeze(1), v).squeeze(1)  # (B, A)
        token = self.proj(pooled)                          # (B, out_dim)
        return token, logits


class DQNet(nn.Module):
    def __init__(self, cfg: DQNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name)
        if cfg.gradient_checkpointing:
            llm.gradient_checkpointing_enable()
            llm.config.use_cache = False

        # Apply LoRA to attention/MLP projections.
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora)
        hidden = llm.config.hidden_size  # Qwen2-0.5B = 896
        self.hidden = hidden

        self.vision_proj = VisionProj(cfg.vision_dim, hidden)
        self.proprio_embed = ProprioEmbed(cfg.proprio_dim, hidden)
        self.action_embed = ActionEmbed(cfg.action_dim, hidden)

        # Supervised target token (method 3): instruction-conditioned attention pool
        # over the agentview patch grid. Query = LLM last-layer hidden state at
        # end of instruction tokens, projected to attn_dim.
        self.target_pool = (
            TargetAttentionPool(cfg.vision_dim, hidden, cfg.target_attn_dim)
            if cfg.use_target_token
            else None
        )
        self.instruction_to_query = (
            nn.Linear(hidden, cfg.target_attn_dim)
            if cfg.use_target_token
            else None
        )

        self.vision_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.vision_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.action_dim),
        )

    # ----- helpers ---------------------------------------------------------
    def _input_embedding_layer(self) -> nn.Module:
        # peft wraps the base model; reach through to the embedding table.
        return self.llm.get_input_embeddings()

    def _embed_token_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self._input_embedding_layer()(ids)

    def _bos_embed(self, batch: int, device, dtype) -> torch.Tensor:
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        ids = torch.full((batch, 1), bos_id, device=device, dtype=torch.long)
        return self._embed_token_ids(ids).to(dtype)

    def _eos_embed(self, batch: int, device, dtype) -> torch.Tensor:
        eos_id = self.tokenizer.eos_token_id
        ids = torch.full((batch, 1), eos_id, device=device, dtype=torch.long)
        return self._embed_token_ids(ids).to(dtype)

    def _tokenize_instructions(self, instructions: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_lang_tokens,
            return_tensors="pt",
        )
        return enc.input_ids.to(device), enc.attention_mask.to(device)

    # ----- main forward ----------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """Compute losses for a training batch."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype  # main model dtype (bf16 if AMP cast)

        instructions: list[str] = batch["instruction"]
        cur_agent = batch["cur_agent"].to(device, dtype=dtype)         # (B, D_v)
        cur_wrist = batch["cur_wrist"].to(device, dtype=dtype)
        cur_proprio = batch["cur_proprio"].to(device, dtype=dtype)
        fut_agent = batch["fut_agent"].to(device, dtype=dtype)         # (B, K, D_v)
        fut_wrist = batch["fut_wrist"].to(device, dtype=dtype)
        fut_actions = batch["fut_actions"].to(device, dtype=dtype)     # (B, K, 7)

        B, K, D_v = fut_agent.shape

        # 1) language tokens
        lang_ids, lang_mask = self._tokenize_instructions(instructions, device)
        lang_embeds = self._embed_token_ids(lang_ids).to(dtype)   # (B, L, H)

        # 2) current observation tokens (3 tokens, or 4 with target token)
        cur_a_emb = self.vision_proj(cur_agent).unsqueeze(1)
        cur_w_emb = self.vision_proj(cur_wrist).unsqueeze(1)
        cur_p_emb = self.proprio_embed(cur_proprio).unsqueeze(1)

        target_attn_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.target_pool is not None:
            cur_patch = batch["cur_agent_patch"].to(device, dtype=dtype)  # (B, N, D_v)
            # --- extract instruction embedding from LLM (first pass) ---
            out_instr = self.llm(
                inputs_embeds=lang_embeds,
                attention_mask=lang_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_instr = out_instr.hidden_states[-1]  # (B, L, H)
            # Take the last instruction token's hidden state as the query
            instruction_embed = hidden_instr[:, -1]  # (B, H)
            instruction_query = self.instruction_to_query(instruction_embed)  # (B, attn_dim)
            # --- build target token ---
            tgt_token, tgt_logits = self.target_pool(cur_patch, instruction_query)  # (B, H), (B, N)
            tgt_emb = tgt_token.unsqueeze(1)
            # current frame: [agent_CLS, agent_TARGET, wrist_CLS, proprio]
            cur_embeds = torch.cat([cur_a_emb, tgt_emb, cur_w_emb, cur_p_emb], dim=1)
            target_attn_loss = self._target_loss(tgt_logits, batch, device)
        else:
            cur_embeds = torch.cat([cur_a_emb, cur_w_emb, cur_p_emb], dim=1)  # (B, 3, H)

        # 3) BOS
        bos_emb = self._bos_embed(B, device, dtype)  # (B, 1, H)

        # 4) future frames: per step k -> [V_agent_{t+k}, V_wrist_{t+k}, ACT_k]
        fut_a_emb = self.vision_proj(fut_agent.reshape(B * K, D_v)).reshape(B, K, -1)
        fut_w_emb = self.vision_proj(fut_wrist.reshape(B * K, D_v)).reshape(B, K, -1)
        act_emb = self.action_embed(fut_actions.reshape(B * K, -1)).reshape(B, K, -1)
        fut_triplets = torch.stack([fut_a_emb, fut_w_emb, act_emb], dim=2)  # (B, K, 3, H)
        fut_embeds = fut_triplets.reshape(B, K * 3, -1)                     # (B, 3K, H)

        # 5) EOS
        eos_emb = self._eos_embed(B, device, dtype)

        # Concat full sequence
        inputs_embeds = torch.cat(
            [lang_embeds, cur_embeds, bos_emb, fut_embeds, eos_emb], dim=1
        )

        # Build attention mask
        L = lang_embeds.size(1)
        cur_len = cur_embeds.size(1)              # 3
        fut_len = fut_embeds.size(1)              # 3K
        suffix_len = 1 + fut_len + 1              # bos + future + eos
        suffix_mask = torch.ones(B, cur_len + suffix_len, dtype=lang_mask.dtype, device=device)
        attn_mask = torch.cat([lang_mask, suffix_mask], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, S, H)

        # ---- locate prediction positions ---------------------------------
        # In the input sequence, frame-k triplet starts at index:
        #   start_k = L + cur_len + 1 + (k-1)*3   (1 for BOS), k in 1..K
        # Causal LMs predict position p+1 from position p, so:
        #   - to predict V_agent_{t+k}: read hidden at (start_k - 1)
        #   - to predict V_wrist_{t+k}: read hidden at  start_k
        #   - to predict ACT_k        : read hidden at (start_k + 1)
        offsets = L + cur_len + 1  # absolute index of the first future-frame triplet
        agent_pred_pos = []
        wrist_pred_pos = []
        action_pred_pos = []
        for k in range(K):
            base = offsets + k * 3
            agent_pred_pos.append(base - 1)
            wrist_pred_pos.append(base)
            action_pred_pos.append(base + 1)
        agent_pos = torch.tensor(agent_pred_pos, device=device)
        wrist_pos = torch.tensor(wrist_pred_pos, device=device)
        action_pos = torch.tensor(action_pred_pos, device=device)

        # Gather hidden states for each prediction site.
        h_agent = hidden_states[:, agent_pos]   # (B, K, H)
        h_wrist = hidden_states[:, wrist_pos]
        h_action = hidden_states[:, action_pos]

        pred_agent = self.vision_head(h_agent)  # (B, K, D_v)
        pred_wrist = self.vision_head(h_wrist)
        pred_action = self.action_head(h_action)  # (B, K, 7)

        # ---- losses ------------------------------------------------------
        cos_a = 1.0 - F.cosine_similarity(pred_agent.float(), fut_agent.float(), dim=-1).mean()
        cos_w = 1.0 - F.cosine_similarity(pred_wrist.float(), fut_wrist.float(), dim=-1).mean()
        vision_loss = 0.5 * (cos_a + cos_w)

        delta_pred = pred_action[..., :6].float()
        delta_gt = fut_actions[..., :6].float()
        action_l1 = F.l1_loss(delta_pred, delta_gt)

        # gripper in {-1, +1} -> map to {0, 1} for BCE
        grip_target = (fut_actions[..., 6:7].float() + 1.0) * 0.5
        grip_logits = pred_action[..., 6:7].float()
        gripper_loss = F.binary_cross_entropy_with_logits(grip_logits, grip_target)

        return {
            "vision_loss": vision_loss,
            "action_loss": action_l1,
            "gripper_loss": gripper_loss,
            "target_loss": target_attn_loss,
            "pred_action": pred_action,
            "pred_agent": pred_agent,
            "pred_wrist": pred_wrist,
        }

    def _target_loss(self, logits: torch.Tensor, batch: dict, device) -> torch.Tensor:
        """Cross-entropy between predicted attention (softmax over patches) and the
        ground-truth target-object heatmap. Samples with no visible target
        (target_valid == False) are masked out. Returns a scalar (0 if none valid)."""
        heat = batch["cur_target_heatmap"].to(device, dtype=torch.float32)  # (B, N)
        valid = batch["cur_target_valid"].to(device).bool().reshape(-1)     # (B,)
        if valid.sum() == 0:
            return torch.zeros((), device=device, dtype=torch.float32)
        logp = F.log_softmax(logits.float(), dim=-1)                        # (B, N)
        # normalize heatmap to a distribution (defensive; extractor already does)
        heat = heat / heat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ce = -(heat * logp).sum(dim=-1)                                     # (B,)
        return ce[valid].mean()

    # ----- inference -------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(
        self,
        instructions: list[str],
        cur_agent: torch.Tensor,
        cur_wrist: torch.Tensor,
        cur_proprio: torch.Tensor,
        cur_agent_patch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Roll out K future frames autoregressively and return action chunk (B, K, 7).

        If the model was built with `use_target_token`, `cur_agent_patch`
        (B, N, vision_dim) must be supplied so the target token can be computed
        by the (already-trained) attention pool. No ground truth is used here.

        The target token's query is extracted from the LLM's last-layer hidden
        state at the end of the instruction tokens (first pass), making the
        attention task-aware.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = cur_agent.size(0)
        K = self.cfg.chunk_size

        cur_agent = cur_agent.to(device, dtype=dtype)
        cur_wrist = cur_wrist.to(device, dtype=dtype)
        cur_proprio = cur_proprio.to(device, dtype=dtype)

        lang_ids, lang_mask = self._tokenize_instructions(instructions, device)
        lang_embeds = self._embed_token_ids(lang_ids).to(dtype)

        cur_a_emb = self.vision_proj(cur_agent).unsqueeze(1).to(dtype)
        cur_w_emb = self.vision_proj(cur_wrist).unsqueeze(1).to(dtype)
        cur_p_emb = self.proprio_embed(cur_proprio).unsqueeze(1).to(dtype)
        if self.target_pool is not None:
            if cur_agent_patch is None:
                raise ValueError(
                    "use_target_token=True requires cur_agent_patch (B, N, vision_dim)"
                )
            # --- extract instruction embedding from LLM (first pass) ---
            out_instr = self.llm(
                inputs_embeds=lang_embeds,
                attention_mask=lang_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_instr = out_instr.hidden_states[-1]  # (B, L, H)
            instruction_embed = hidden_instr[:, -1]  # (B, H)
            instruction_query = self.instruction_to_query(instruction_embed)  # (B, attn_dim)
            # --- build target token ---
            cur_patch = cur_agent_patch.to(device, dtype=dtype)
            tgt_token, _ = self.target_pool(cur_patch, instruction_query)
            tgt_emb = tgt_token.unsqueeze(1).to(dtype)
            cur_embeds = torch.cat([cur_a_emb, tgt_emb, cur_w_emb, cur_p_emb], dim=1)
        else:
            cur_embeds = torch.cat([cur_a_emb, cur_w_emb, cur_p_emb], dim=1)
        bos_emb = self._bos_embed(B, device, dtype)

        seq = torch.cat([lang_embeds, cur_embeds, bos_emb], dim=1)
        seq_mask = torch.cat(
            [lang_mask, torch.ones(B, cur_embeds.size(1) + 1, dtype=lang_mask.dtype, device=device)],
            dim=1,
        )

        actions_out: list[torch.Tensor] = []
        for _ in range(K):
            out = self.llm(
                inputs_embeds=seq,
                attention_mask=seq_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h = out.hidden_states[-1]
            # next token to be produced is V_agent of upcoming future frame.
            # We read its prediction from the last hidden state.
            v_agent_pred_feat = self.vision_head(h[:, -1])           # (B, D_v)
            v_agent_emb = self.vision_proj(v_agent_pred_feat).unsqueeze(1).to(dtype)
            seq = torch.cat([seq, v_agent_emb], dim=1)
            seq_mask = torch.cat(
                [seq_mask, torch.ones(B, 1, dtype=seq_mask.dtype, device=device)], dim=1
            )

            out = self.llm(
                inputs_embeds=seq,
                attention_mask=seq_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h = out.hidden_states[-1]
            v_wrist_pred_feat = self.vision_head(h[:, -1])
            v_wrist_emb = self.vision_proj(v_wrist_pred_feat).unsqueeze(1).to(dtype)
            seq = torch.cat([seq, v_wrist_emb], dim=1)
            seq_mask = torch.cat(
                [seq_mask, torch.ones(B, 1, dtype=seq_mask.dtype, device=device)], dim=1
            )

            out = self.llm(
                inputs_embeds=seq,
                attention_mask=seq_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h = out.hidden_states[-1]
            action_pred = self.action_head(h[:, -1])  # (B, 7)
            # convert gripper logit to discrete -1/+1 sign
            grip = (torch.sigmoid(action_pred[..., 6:7]) > 0.5).float() * 2.0 - 1.0
            action_decoded = torch.cat([action_pred[..., :6], grip], dim=-1)
            actions_out.append(action_decoded)

            act_emb = self.action_embed(action_decoded.to(dtype)).unsqueeze(1).to(dtype)
            seq = torch.cat([seq, act_emb], dim=1)
            seq_mask = torch.cat(
                [seq_mask, torch.ones(B, 1, dtype=seq_mask.dtype, device=device)], dim=1
            )

        return torch.stack(actions_out, dim=1)  # (B, K, 7)


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
