from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_HEAD_WEIGHTS_NAME = "latent_attention_head.pt"


def _resolve_latent_dim(hidden_size: int, latent_dim: int) -> int:
    if latent_dim is None or int(latent_dim) <= 0:
        return int(hidden_size)
    return int(latent_dim)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if dim_head <= 0:
            raise ValueError(f"dim_head must be positive, got {dim_head}")

        inner_dim = heads * dim_head
        self.heads = int(heads)
        self.dim_head = int(dim_head)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

    def _build_sdpa_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # 1 -> keep token, 0 -> pad token. SDPA additive mask: keep=0, pad=-1e4.
        keep = attention_mask.to(dtype=dtype)
        return (1.0 - keep).unsqueeze(1).unsqueeze(2) * -10000.0

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, query_len, _ = x.shape
        _, key_len, _ = context.shape

        q = self.to_q(x).view(bsz, query_len, self.heads, self.dim_head).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(bsz, key_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(bsz, key_len, self.heads, self.dim_head).transpose(1, 2)

        sdpa_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(f"attention_mask must be [B, L], got {tuple(attention_mask.shape)}")
            sdpa_mask = self._build_sdpa_mask(attention_mask, dtype=q.dtype)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, query_len, self.heads * self.dim_head)
        return self.to_out(out)


class LatentAttentionPooling(nn.Module):
    """
    Single-layer latent-attention pooling:
    learnable latent queries -> cross-attend token states -> FFN -> mean over latent slots.
    """

    def __init__(
        self,
        hidden_size: int,
        num_latents: int = 128,
        latent_dim: int = -1,
        num_cross_heads: int = 8,
        cross_dim_head: int = 64,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if num_latents <= 0:
            raise ValueError(f"num_latents must be positive, got {num_latents}")

        self.hidden_size = int(hidden_size)
        self.num_latents = int(num_latents)
        self.latent_dim = _resolve_latent_dim(hidden_size=self.hidden_size, latent_dim=latent_dim)
        self.num_cross_heads = int(num_cross_heads)
        self.cross_dim_head = int(cross_dim_head)
        self.ff_mult = int(ff_mult)
        self.readout_type = "mean"

        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        self.cross_attn_norm = nn.LayerNorm(self.latent_dim)
        self.cross_ctx_norm = nn.LayerNorm(self.hidden_size)
        self.cross_attn = Attention(
            query_dim=self.latent_dim,
            context_dim=self.hidden_size,
            heads=self.num_cross_heads,
            dim_head=self.cross_dim_head,
        )
        self.ff_norm = nn.LayerNorm(self.latent_dim)
        self.ff = FeedForward(self.latent_dim, mult=self.ff_mult)
        self.out_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz = hidden_states.size(0)
        latent_tokens = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        attn_out = self.cross_attn(
            self.cross_attn_norm(latent_tokens),
            context=self.cross_ctx_norm(hidden_states),
            attention_mask=attention_mask,
        )
        latent_tokens = latent_tokens + attn_out
        latent_tokens = latent_tokens + self.ff(self.ff_norm(latent_tokens))
        latent_tokens = self.out_norm(latent_tokens)
        return latent_tokens.mean(dim=1)

    def export_config(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "num_latents": self.num_latents,
            "latent_dim": self.latent_dim,
            "num_cross_heads": self.num_cross_heads,
            "cross_dim_head": self.cross_dim_head,
            "ff_mult": self.ff_mult,
            "readout_type": self.readout_type,
        }


def _infer_layer_indices(state_dict: Dict[str, torch.Tensor]) -> set[int]:
    layer_indices = set()
    for key in state_dict.keys():
        if not key.startswith("layers."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_indices.add(int(parts[1]))
        except ValueError:
            continue
    return layer_indices


def _flatten_single_layer_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    layer_indices = _infer_layer_indices(state_dict)
    if not layer_indices:
        return state_dict

    if layer_indices != {0}:
        raise ValueError(
            f"Detected deprecated multi-layer latent head weights with layers={sorted(layer_indices)}. "
            "Current code only supports single-layer latent attention."
        )

    flattened: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("layers.0."):
            flattened[key[len("layers.0."):]] = value
        elif key.startswith("layers."):
            # Already validated only layer 0 is allowed.
            continue
        else:
            flattened[key] = value
    return flattened


def save_latent_attention_head(
    output_dir: str,
    head_module: LatentAttentionPooling,
    config: Dict[str, int],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, LATENT_HEAD_WEIGHTS_NAME)
    payload = {
        "format_version": 1,
        "config": dict(config),
        "state_dict": {k: v.detach().cpu() for k, v in head_module.state_dict().items()},
    }
    torch.save(payload, output_file)
    return output_file


def load_latent_attention_head(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> Tuple[Dict[str, int], Dict[str, torch.Tensor], str]:
    if os.path.isdir(checkpoint_path):
        file_path = os.path.join(checkpoint_path, LATENT_HEAD_WEIGHTS_NAME)
    else:
        file_path = checkpoint_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"latent attention head not found: {file_path}")

    payload = torch.load(file_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid latent head payload type: {type(payload)}")

    if "state_dict" in payload:
        config = payload.get("config", {})
        state_dict = payload["state_dict"]
    else:
        config = {}
        state_dict = payload

    if not isinstance(state_dict, dict):
        raise ValueError("Invalid latent head state_dict payload.")

    if int(config.get("num_layers", 1) or 1) > 1:
        raise ValueError(
            f"Deprecated latent head config with num_layers={config.get('num_layers')} is not supported."
        )
    state_dict = _flatten_single_layer_state_dict(state_dict)

    return dict(config), state_dict, file_path
