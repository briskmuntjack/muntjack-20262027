"""x-prediction MeanFlow (x-MF) generator.

The network predicts a clean latent x_hat(z_t, r, t | c, omega) from a noisy
latent z_t, the current time t, the target time r and the guidance scale omega.
Moving from t to r is then one closed-form update (Eq. "xpred_step" in the
paper):

    z_r = (r / t) z_t + (1 - r / t) x_hat(z_t, r, t | c, omega)

so a K-step sample costs exactly K network evaluations, and K = 1 is a direct
map from the initial noise to the clean latent. That one-step map is what the
Single-Step Search previews and what the Noise Booster is trained through.

Architecture: the TangoFlux text-to-audio backbone (a Flux transformer with
6 joint and 18 single blocks, 64-channel audio latents of 215 frames for 10 s,
a frozen FLAN-T5-large text encoder, a duration embedding), plus a zero-init
guidance-scale (omega) embedder added at iMF fine-tuning. The guidance scale is
a network input, so sampling never needs a separate unconditional pass.
"""

from math import pi
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn
from diffusers import FluxTransformer2DModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers import T5EncoderModel, T5TokenizerFast

__all__ = ["XMFGenerator", "OmegaEmbedder", "DurationEmbedder"]


class OmegaEmbedder(nn.Module):
    """Guidance-scale conditioning.

    omega is scaled by 1000 and passed through a sinusoidal embedding and a
    small MLP, like the Flux timestep/guidance inputs. The output is added to
    the pooled text projection that modulates the transformer. The last linear
    layer is zero-initialised so that a freshly created embedder is a no-op.
    """

    def __init__(self, dim_out: int, freq_channels: int = 256):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=freq_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.embedder = TimestepEmbedding(in_channels=freq_channels, time_embed_dim=dim_out)
        nn.init.zeros_(self.embedder.linear_2.weight)
        nn.init.zeros_(self.embedder.linear_2.bias)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        emb = self.time_proj(omega.to(torch.float32) * 1000.0)
        return self.embedder(emb.to(omega.dtype) if omega.dtype != torch.float32 else emb)


class _PositionalEmbedding(nn.Module):
    """Learned-frequency Fourier features for a scalar (from Stable Audio Open)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((times, fouriered), dim=-1)


class DurationEmbedder(nn.Module):
    """Maps the requested duration (seconds) to one conditioning token."""

    def __init__(self, number_embedding_dim: int, min_value: float, max_value: float,
                 internal_dim: int = 256):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            _PositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )
        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, floats: torch.Tensor) -> torch.Tensor:
        floats = floats.clamp(self.min_value, self.max_value)
        normalized = (floats - self.min_value) / (self.max_value - self.min_value)
        dtype = next(self.time_positional_embedding.parameters()).dtype
        embedding = self.time_positional_embedding(normalized.to(dtype))
        return embedding.view(-1, 1, self.number_embedding_dim)


class XMFGenerator(nn.Module):
    """x-MF generator: text/duration conditioning, the x-prediction network and
    the K-step sampler."""

    def __init__(self, config: dict, text_encoder_dir: Optional[str] = None):
        super().__init__()
        self.num_layers = config.get("num_layers", 6)
        self.num_single_layers = config.get("num_single_layers", 18)
        self.in_channels = config.get("in_channels", 64)
        self.attention_head_dim = config.get("attention_head_dim", 128)
        self.joint_attention_dim = config.get("joint_attention_dim", 1024)
        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.audio_seq_len = config.get("audio_seq_len", 215)
        self.max_duration = config.get("max_duration", 10)
        self.text_encoder_name = config.get("text_encoder_name", "google/flan-t5-large")
        self.omega_cond = bool(config.get("omega_cond", True))
        self.max_text_seq_len = 64

        text_src = text_encoder_dir or self.text_encoder_name
        self.text_encoder = T5EncoderModel.from_pretrained(text_src)
        self.tokenizer = T5TokenizerFast.from_pretrained(text_src)
        self.text_embedding_dim = self.text_encoder.config.d_model

        self.fc = nn.Sequential(
            nn.Linear(self.text_embedding_dim, self.joint_attention_dim), nn.ReLU()
        )
        self.duration_embedder = DurationEmbedder(
            self.text_embedding_dim, min_value=0, max_value=self.max_duration
        )
        ## guidance_embeds=True: the target time r enters through the guidance
        ## embedding of the Flux transformer (scaled by 1000 internally, like t).
        self.transformer = FluxTransformer2DModel(
            in_channels=self.in_channels,
            num_layers=self.num_layers,
            num_single_layers=self.num_single_layers,
            attention_head_dim=self.attention_head_dim,
            num_attention_heads=self.num_attention_heads,
            joint_attention_dim=self.joint_attention_dim,
            pooled_projection_dim=self.text_embedding_dim,
            guidance_embeds=True,
        )
        self.omega_embedder = (
            OmegaEmbedder(dim_out=self.joint_attention_dim) if self.omega_cond else None
        )

    # ------------------------------------------------------------ conditioning
    @property
    def device(self) -> torch.device:
        return self.transformer.device

    @torch.no_grad()
    def encode_text(self, prompts: List[str]):
        batch = self.tokenizer(
            prompts, max_length=self.max_text_seq_len, padding=True, truncation=True,
            return_tensors="pt",
        )
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return hidden, (attention_mask == 1)

    @torch.no_grad()
    def build_cond(
        self,
        prompts: Union[str, Sequence[str]],
        durations: Union[float, Sequence[float], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Conditioning dict for `predict_x` / `sample` (and for the Noise Booster).

        encoder_hidden_states : T5 token states followed by the duration token
        pooled_projection     : fc(mean of the valid T5 token states), without omega
        txt_ids / img_ids     : position ids expected by the Flux transformer
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(
                [durations] if not isinstance(durations, (list, tuple)) else list(durations),
                dtype=torch.float32,
            )
        durations = durations.to(self.device).float().clamp(max=float(self.max_duration))
        if durations.shape[0] != len(prompts):
            durations = durations.expand(len(prompts))

        hidden, mask = self.encode_text(prompts)
        duration_states = self.duration_embedder(durations)

        bsz = hidden.shape[0]
        masked = torch.where(
            mask.unsqueeze(-1).expand_as(hidden), hidden,
            torch.tensor(float("nan"), device=hidden.device),
        )
        pooled_projection = self.fc(torch.nanmean(masked, dim=1))
        encoder_hidden_states = torch.cat([hidden, duration_states], dim=1)

        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3, device=self.device)
        img_ids = (
            torch.arange(self.audio_seq_len).unsqueeze(0).unsqueeze(-1)
            .repeat(bsz, 1, 3).to(self.device)
        )
        return {
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projection": pooled_projection,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
        }

    def omega_tensor(self, batch_size: int, omega: Optional[float]) -> Optional[torch.Tensor]:
        if self.omega_embedder is None:
            return None
        return torch.full((batch_size,), float(omega), device=self.device, dtype=torch.float32)

    # ----------------------------------------------------------------- network
    def predict_x(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        omega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x_hat(z_t, r, t | c, omega).  z_t: (B, 1, T, C) -> (B, 1, T, C).

        r, t: (B,) in [0, 1]. t is the timestep input, r the guidance input.
        omega: (B,) guidance scales added through the omega embedder.
        """
        pooled = cond["pooled_projection"]
        if omega is not None and self.omega_embedder is not None:
            pooled = pooled + self.omega_embedder(omega)
        x_pred = self.transformer(
            hidden_states=z_t.squeeze(1),
            timestep=t,
            guidance=r,
            pooled_projections=pooled,
            encoder_hidden_states=cond["encoder_hidden_states"],
            txt_ids=cond["txt_ids"],
            img_ids=cond["img_ids"],
            return_dict=False,
        )[0]
        return x_pred.unsqueeze(1)

    # ----------------------------------------------------------------- sampler
    def sample(
        self,
        cond: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        num_steps: int,
        omega: Optional[float] = 3.5,
    ) -> torch.Tensor:
        """K-step x-MF sampling from `noise` (B, 1, T, C) -> latents (B, T, C).

        Uniform grid t_k = 1 - k/K. Each step evaluates the network once at
        (t_k, t_{k+1}) and applies the x-prediction update. With num_steps=1
        the result is the network output itself, x_hat(z, 0, 1 | c, omega).
        Differentiable (no `no_grad` here): the Noise Booster is trained through
        the one-step call. Wrap in `torch.no_grad()` for plain inference.
        """
        bsz = noise.shape[0]
        expected = (bsz, 1, self.audio_seq_len, self.in_channels)
        if tuple(noise.shape) != expected:
            raise ValueError(f"noise has shape {tuple(noise.shape)}, expected {expected}")
        x = noise.to(self.device)
        omega_t = self.omega_tensor(bsz, omega)
        grid = torch.linspace(1, 0, num_steps + 1, device=x.device, dtype=x.dtype)
        for k in range(num_steps):
            t = grid[k].expand(bsz)
            r = grid[k + 1].expand(bsz)
            x_pred = self.predict_x(x, r, t, cond, omega=omega_t)
            v_pred = (x_pred - x) / t.reshape(bsz, -1, 1, 1)
            x = x + (t - r).mean() * v_pred
        return x.squeeze(1)

    # ------------------------------------------------------------- checkpoint
    def load_weights(self, state_dict: Dict[str, torch.Tensor], verbose: bool = True):
        """Load released weights. Text-encoder weights are not part of the
        checkpoint (FLAN-T5 is frozen and loaded from the hub), so only those
        keys may be missing."""
        state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            ## key spelling used by earlier training checkpoints
            if k.startswith("duration_emebdder."):
                k = "duration_embedder." + k[len("duration_emebdder."):]
            state[k] = v
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("text_encoder.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match the x-MF generator: "
                f"missing {bad_missing[:5]}, unexpected {list(unexpected)[:5]}"
            )
        if verbose:
            n = sum(v.numel() for k, v in state.items() if not k.startswith("text_encoder."))
            print(f"[x-MF] loaded {n / 1e6:.1f}M generator parameters")
        return missing, unexpected
