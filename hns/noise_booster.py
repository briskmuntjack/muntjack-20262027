"""Noise Booster B_phi: prompt-conditioned refinement of the initial noise.

    z' = z + B_phi(z, c)

A lightweight DiT: learned positional embeddings on the noise frames, then a
stack of blocks with self-attention over the latent frames, cross-attention to
the text tokens, and an MLP, each modulated by adaLN-Zero from the pooled text
embedding of the generator. There is no diffusion timestep: the booster refines
the initial noise directly. The output projection and all adaLN layers are
zero-initialised, so an untrained booster is the identity (z' = z).

The conditioning dict is the one `XMFGenerator.build_cond` produces, so the
booster reuses the generator's text encoder and pooled projection.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NoiseBooster"]


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _Attention(nn.Module):
    """Self-attention over latent frames, or cross-attention to text tokens."""

    def __init__(self, dim: int, heads: int, kv_dim: int = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        kv_dim = kv_dim or dim
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(kv_dim, dim, bias=True)
        self.to_v = nn.Linear(kv_dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x, kv=None, kv_mask=None):
        kv = x if kv is None else kv
        b, n, _ = x.shape
        m = kv.shape[1]
        q = self.to_q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(kv).view(b, m, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(kv).view(b, m, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask[:, None, None, :].expand(b, self.heads, n, m)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class _Block(nn.Module):
    """DiT block: self-attn -> cross-attn -> MLP, each adaLN-Zero modulated."""

    def __init__(self, dim: int, heads: int, text_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross = _Attention(dim, heads, kv_dim=text_dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, text, text_mask):
        (sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3) = self.ada(c).chunk(9, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), sh1, sc1))
        x = x + g2.unsqueeze(1) * self.cross(
            _modulate(self.norm2(x), sh2, sc2), kv=text, kv_mask=text_mask
        )
        x = x + g3.unsqueeze(1) * self.mlp(_modulate(self.norm3(x), sh3, sc3))
        return x


class NoiseBooster(nn.Module):
    """(z, cond) -> residual update delta, same shape as z."""

    def __init__(
        self,
        in_channels: int = 64,
        seq_len: int = 216,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        text_dim: int = 1024,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.x_proj = nn.Linear(in_channels, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.normal_(self.pos, std=0.02)

        ## global conditioning from the generator's pooled text projection
        self.c_proj = nn.Sequential(nn.Linear(text_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [_Block(dim, heads, text_dim, mlp_ratio) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.out = nn.Linear(dim, in_channels)
        ## identity at initialisation
        nn.init.zeros_(self.ada_out[1].weight)
        nn.init.zeros_(self.ada_out[1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @classmethod
    def from_config(cls, cfg: dict) -> "NoiseBooster":
        return cls(
            in_channels=int(cfg.get("in_channels", 64)),
            seq_len=int(cfg.get("seq_len", 216)),
            dim=int(cfg.get("dim", 512)),
            depth=int(cfg.get("depth", 6)),
            heads=int(cfg.get("heads", 8)),
            text_dim=int(cfg.get("text_dim", 1024)),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        )

    def forward(self, z: torch.Tensor, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        """z (B, 1, T, C) -> delta (B, 1, T, C).

        T may be shorter than `seq_len` (the released booster was trained on
        216 frames and is applied to the generator's 215); the first T
        positional entries are used.
        """
        x = self.x_proj(z.squeeze(1))
        t = x.shape[1]
        if t > self.seq_len:
            raise ValueError(f"{t} frames, but the booster was built for {self.seq_len}")
        x = x + self.pos[:, :t]

        c = self.c_proj(cond["pooled_projection"])
        text = cond["encoder_hidden_states"]
        text_mask = cond.get("encoder_attention_mask")
        for blk in self.blocks:
            x = blk(x, c, text, text_mask)

        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        x = _modulate(self.norm_out(x), shift, scale)
        return self.out(x).unsqueeze(1)

    def boost(self, z: torch.Tensor, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        """z' = z + B_phi(z, c)."""
        return z + self.forward(z, cond)

    def load_weights(self, state_dict: Dict[str, torch.Tensor], verbose: bool = True):
        state = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state_dict.items()}
        self.load_state_dict(state, strict=True)
        if verbose:
            n = sum(p.numel() for p in self.parameters())
            print(f"[Noise Booster] loaded {n / 1e6:.1f}M parameters")
