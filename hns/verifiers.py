"""Verifiers used by the Single-Step Search (and as rewards for the Noise Booster).

  CLAPVerifier        text-audio relevance: LAION-CLAP cosine similarity
                      (`music_speech_audioset_epoch_15_esc_89.98.pt`, 48 kHz mono)
  AestheticsVerifier  perceptual quality: Audiobox Aesthetics axes, PQ is the
                      one used for selection (16 kHz mono)

Both are reference-free: they score a generated clip against the prompt (CLAP)
or on its own (PQ), so no ground-truth audio is involved at inference.

Preprocessing follows the av-benchmark evaluation path (mono by channel mean,
DC removal and 48 kHz for CLAP; mono and 16 kHz for Audiobox Aesthetics), so
the numbers are on the same scale as the CLAP / PQ columns of the paper.
"""

import os
from typing import Dict, List, Optional, Sequence

import torch
import torchaudio

__all__ = ["CLAP_SR", "AES_SR", "AES_AXES", "CLAPVerifier", "AestheticsVerifier"]

CLAP_SR = 48000
AES_SR = 16000
AES_AXES = ("PQ", "CE", "CU", "PC")

## CLAP's log-mel front end is slightly batch-shape sensitive (the STFT is a
## cuDNN conv whose algorithm depends on the batch size). The chunk size is
## therefore kept fixed at the value used for every number in the paper.
DEFAULT_CLAP_CHUNK = 16


class CLAPVerifier:
    """R_clap(a, c): LAION-CLAP cosine similarity between audio and caption."""

    def __init__(self, ckpt: str, device: torch.device, audio_length: float = 10.0):
        import laion_clap

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"LAION-CLAP checkpoint not found: {ckpt}\n"
                "Download music_speech_audioset_epoch_15_esc_89.98.pt from "
                "https://huggingface.co/lukewys/laion_clap and pass its path."
            )
        self.device = device
        self.audio_length = float(audio_length)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(ckpt, verbose=False)
        self.clap = self.clap.to(device).eval()
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def text_embedding(self, caption: str) -> torch.Tensor:
        if caption not in self._text_cache:
            emb = self.clap.get_text_embedding([caption], use_tensor=True)
            self._text_cache[caption] = emb.float().to(self.device)
        return self._text_cache[caption]

    @torch.no_grad()
    def score(
        self,
        wav: torch.Tensor,
        sr: int,
        caption: str,
        chunk: int = DEFAULT_CLAP_CHUNK,
    ) -> torch.Tensor:
        """wav (B, ch, n) -> (B,) cosine similarities (on CPU)."""
        text_emb = self.text_embedding(caption)
        mono = wav.mean(dim=1).to(self.device).float()
        mono = mono - mono.mean(dim=1, keepdim=True)
        if sr != CLAP_SR:
            mono = torchaudio.functional.resample(mono, sr, CLAP_SR)
        mono = mono[:, : int(CLAP_SR * self.audio_length)]
        n = mono.shape[0]
        step = n if not chunk or chunk >= n else chunk
        out = []
        for i in range(0, n, step):
            emb = self.clap.get_audio_embedding_from_data(mono[i:i + step], use_tensor=True).float()
            out.append(torch.cosine_similarity(emb, text_emb.expand_as(emb), dim=1).cpu())
        return torch.cat(out)


class AestheticsVerifier:
    """R_pq(a): Audiobox Aesthetics production quality (plus CE / CU / PC)."""

    def __init__(self, ckpt: Optional[str], device: torch.device):
        from audiobox_aesthetics.infer import initialize_predictor

        self.device = device
        ## None -> the official checkpoint is fetched from the Hugging Face hub.
        self.predictor = initialize_predictor(ckpt if ckpt and os.path.exists(ckpt) else None)

    @torch.no_grad()
    def score(
        self,
        wav: torch.Tensor,
        sr: int,
        chunk: int = DEFAULT_CLAP_CHUNK,
        axes: Sequence[str] = AES_AXES,
    ) -> Dict[str, torch.Tensor]:
        """wav (B, ch, n) -> {axis: (B,)} on the original 1-10 scale (on CPU)."""
        mono = wav.mean(dim=1, keepdim=True).float().cpu()
        if sr != AES_SR:
            mono = torchaudio.functional.resample(mono, sr, AES_SR)
        n = mono.shape[0]
        step = n if not chunk or chunk >= n else chunk
        rows: List[dict] = []
        for i in range(0, n, step):
            rows += self.predictor.forward(
                [{"path": w, "sample_rate": AES_SR} for w in mono[i:i + step]]
            )
        return {a: torch.tensor([r[a] for r in rows]) for a in axes}
