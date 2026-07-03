# GlanceVLA Method Detail

## Supervised Target Token

### TargetAttentionPool

A single learnable query over G×G pooled DINOv2 patches:

```python
class TargetAttentionPool(nn.Module):
    def __init__(self, vision_dim, out_dim, attn_dim=256):
        self.norm  = LayerNorm(vision_dim)
        self.query = Parameter(torch.randn(1, 1, attn_dim) * 0.02)
        self.to_k  = Linear(vision_dim, attn_dim)
        self.to_v  = Linear(vision_dim, attn_dim)
        self.proj  = MLP(attn_dim, out_dim)

    def forward(self, patches):
        x = self.norm(patches)
        k, v = self.to_k(x), self.to_v(x)
        q = self.query.expand(x.size(0), -1, -1)
        logits = (q @ k.transpose(1, 2)) * (attn_dim ** -0.5)
        weights = softmax(logits, dim=-1)
        return self.proj(weights @ v), logits
```

### Supervision Signal

From `extract_features_target.py`:
1. Render frames using `SegmentationRenderEnv` with instance segmentation
2. Extract `obj_of_interest` mask from `env.instance_to_id[obj]`
3. Vertical flip (matches encoder) → adaptive pool to G×G → normalize to sum=1
4. Empty mask → `target_valid=False` → frame excluded from loss

### Loss

```python
ce = -(heatmap * log_softmax(logits)).sum(-1)
target_loss = ce[valid].mean()
```

## Token Sequence

Only current frame changes (3→4 tokens). Future-frame triplets and all action-decoding positions are unchanged. The offset `offsets = L + cur_len + 1` derives from `cur_embeds.size(1)`.

## Engineering Notes

- **exec_horizon=1 required** — full-chunk autoregressive rollout causes drift accumulation
- **Loss floor ~1.0** — target term has entropy lower bound (~1.73); monitor action/vision losses for training progress
- **Target token uses union of obj_of_interest** — edit `target_mask()` to focus on single manipuland
