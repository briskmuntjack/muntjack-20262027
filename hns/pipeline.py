"""Hybrid Noise Skimming inference (Algorithm 1 of the paper).

    1. (N, K) <- budget(nfe)
    2. draw N Gaussian noises z_i
    3. Noise Booster:        z_i' = z_i + B_phi(z_i, c)
    4. Single-Step Search:   if N > 1, preview a_i = D(G^[1](z_i'; c)), score with
                             CLAP and PQ, pick i* by rank-mean
    5. final generation:     a* = D(G^[K](z_{i*}'; c))

Everything runs in float32 on one GPU.
"""

import os
from typing import Dict, Optional, Tuple, Union

import torch
from diffusers import AutoencoderOobleck

from .noise_booster import NoiseBooster
from .search import budget, select_best
from .utils import HF_REPO, REPO_ROOT, gaussian_noise, load_safetensors, load_yaml, resolve_checkpoint
from .verifiers import AES_AXES, DEFAULT_CLAP_CHUNK, AestheticsVerifier, CLAPVerifier
from .xmf import XMFGenerator

__all__ = ["HNSPipeline", "GENERATOR_FILE", "BOOSTER_FILE", "CLAP_FILE", "AES_FILE"]

GENERATOR_FILE = "xmf_generator.safetensors"
BOOSTER_FILE = "noise_booster.safetensors"
CLAP_FILE = "music_speech_audioset_epoch_15_esc_89.98.pt"
AES_FILE = "audiobox_aesthetics_checkpoint.pt"
DEFAULT_VAE = "stabilityai/stable-audio-open-1.0"
## Hub repository of the LAION-CLAP verifier checkpoint.
CLAP_HF_REPO = "lukewys/laion_clap"


def _expand_cond(cond: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    """A single-prompt conditioning dict repeated for n candidates."""
    return {k: v.expand(n, *v.shape[1:]) for k, v in cond.items()}


class HNSPipeline:
    def __init__(
        self,
        generator: XMFGenerator,
        vae: AutoencoderOobleck,
        booster: Optional[NoiseBooster] = None,
        clap: Optional[CLAPVerifier] = None,
        aesthetics: Optional[AestheticsVerifier] = None,
        omega: float = 3.5,
        final_steps: int = 2,
        search_steps: int = 1,
        max_duration: float = 10.0,
    ):
        self.generator = generator.eval()
        self.vae = vae.eval()
        self.booster = booster.eval() if booster is not None else None
        self.clap = clap
        self.aesthetics = aesthetics
        self.omega = float(omega)
        self.final_steps = int(final_steps)
        self.search_steps = int(search_steps)
        self.max_duration = float(max_duration)
        self.sample_rate = int(vae.config.sampling_rate)

    @property
    def device(self) -> torch.device:
        return self.generator.device

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str = os.path.join(REPO_ROOT, "checkpoints"),
        config_dir: str = os.path.join(REPO_ROOT, "configs"),
        vae: str = DEFAULT_VAE,
        device: Union[str, torch.device] = "cuda",
        use_booster: bool = True,
        use_verifiers: bool = True,
        clap_ckpt: Optional[str] = None,
        aes_ckpt: Optional[str] = None,
        text_encoder_dir: Optional[str] = None,
        hf_repo: Optional[str] = HF_REPO,
    ) -> "HNSPipeline":
        """Build the pipeline from `checkpoint_dir`:

            xmf_generator.safetensors                      x-MF generator
            noise_booster.safetensors                      Noise Booster
            music_speech_audioset_epoch_15_esc_89.98.pt    LAION-CLAP (verifier)
            audiobox_aesthetics_checkpoint.pt              optional; fetched from the
                                                           hub when absent

        Files missing from `checkpoint_dir` are downloaded into it on first use:
        the generator and the Noise Booster from `hf_repo`
        (briskmuntjack/hybrid-noise-skimming), the CLAP checkpoint from
        lukewys/laion_clap. `hf_repo=None` disables the download of the two
        released models.

        `vae` is either a local directory holding the Stable Audio Open VAE or
        the hub id `stabilityai/stable-audio-open-1.0` (subfolder `vae`).
        """
        device = torch.device(device)
        gcfg = load_yaml(os.path.join(config_dir, "xmf.yaml"))
        bcfg = load_yaml(os.path.join(config_dir, "noise_booster.yaml"))
        sampling = gcfg.get("sampling", {})

        generator = XMFGenerator(gcfg["model"], text_encoder_dir=text_encoder_dir)
        generator.load_weights(load_safetensors(resolve_checkpoint(checkpoint_dir, GENERATOR_FILE, hf_repo)))
        generator.to(device).eval()

        if os.path.isdir(vae):
            vae_model = AutoencoderOobleck.from_pretrained(vae)
        else:
            vae_model = AutoencoderOobleck.from_pretrained(vae, subfolder="vae")
        vae_model.to(device).eval()

        booster = None
        if use_booster:
            booster = NoiseBooster.from_config(bcfg["model"])
            booster.load_weights(load_safetensors(resolve_checkpoint(checkpoint_dir, BOOSTER_FILE, hf_repo)))
            booster.to(device).eval()

        clap = aesthetics = None
        if use_verifiers:
            max_dur = float(sampling.get("max_audio_duration", 10.0))
            clap = CLAPVerifier(clap_ckpt or resolve_checkpoint(checkpoint_dir, CLAP_FILE, CLAP_HF_REPO),
                                device, max_dur)
            aes_path = aes_ckpt or os.path.join(checkpoint_dir, AES_FILE)
            aesthetics = AestheticsVerifier(aes_path if os.path.exists(aes_path) else None, device)

        return cls(
            generator, vae_model, booster, clap, aesthetics,
            omega=float(sampling.get("omega", 3.5)),
            final_steps=int(sampling.get("final_steps", 2)),
            search_steps=int(sampling.get("search_steps", 1)),
            max_duration=float(sampling.get("max_audio_duration", 10.0)),
        )

    # ------------------------------------------------------------- generation
    def sample_noise(self, num: int, seed: int = 0) -> torch.Tensor:
        return gaussian_noise(num, self.generator.audio_seq_len, self.generator.in_channels, seed)

    @torch.no_grad()
    def _generate(
        self,
        cond1: Dict[str, torch.Tensor],
        z: torch.Tensor,
        num_steps: int,
        gen_batch_size: int = 0,
        decode_batch_size: int = 0,
    ) -> torch.Tensor:
        """Generate and decode every noise in z (N, 1, T, C) -> waves (N, 2, n) on CPU."""
        n = z.shape[0]
        gen_bs = gen_batch_size or n
        dec_bs = decode_batch_size or n
        waves = []
        for i in range(0, n, gen_bs):
            zb = z[i:i + gen_bs].to(self.device)
            lat = self.generator.sample(_expand_cond(cond1, zb.shape[0]), zb, num_steps, self.omega)
            for j in range(0, lat.shape[0], dec_bs):
                waves.append(self.vae.decode(lat[j:j + dec_bs].transpose(2, 1)).sample.cpu())
        return torch.cat(waves)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        nfe: int = 16,
        duration: float = 10.0,
        seed: int = 0,
        noise: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        num_steps: Optional[int] = None,
        use_booster: bool = True,
        rule: str = "rank_mean",
        gen_batch_size: int = 8,
        decode_batch_size: int = 2,
        clap_chunk: int = DEFAULT_CLAP_CHUNK,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Text -> stereo waveform (2, n) at 44.1 kHz, float32 on the CPU.

        nfe            total generator budget; mapped to (N candidates, K final
                       steps) by the paper's rule unless `num_candidates` /
                       `num_steps` are given explicitly
        duration       requested length in seconds (<= 10)
        seed           seed of the N Gaussian candidates (ignored when `noise` is given)
        noise          (N, 1, 215, 64) standard-normal candidates to start from
        use_booster    apply the Noise Booster before the search / generation
        rule           'rank_mean' (CLAP + PQ, paper default) | 'clap' | 'pq'
        gen_batch_size / decode_batch_size
                       candidates per generator forward / VAE decode during the
                       search (0 = all at once; faster, more memory)
        return_info    also return the per-candidate verifier scores and the
                       selected index
        """
        n_cand, k_steps = budget(nfe, self.final_steps)
        if num_candidates is not None:
            n_cand = int(num_candidates)
        if num_steps is not None:
            k_steps = int(num_steps)
        if n_cand > 1 and (self.clap is None or self.aesthetics is None):
            raise RuntimeError("Single-Step Search needs the CLAP and Audiobox verifiers "
                               "(build the pipeline with use_verifiers=True)")
        duration = min(float(duration), self.max_duration)

        z = noise if noise is not None else self.sample_noise(n_cand, seed)
        if z.shape[0] != n_cand:
            raise ValueError(f"noise holds {z.shape[0]} candidates but N = {n_cand}")
        z = z.to(self.device)

        cond1 = self.generator.build_cond([prompt], [duration])
        if use_booster and self.booster is not None:
            z = self.booster.boost(z, _expand_cond(cond1, n_cand))

        info = {"nfe": nfe, "num_candidates": n_cand, "final_steps": k_steps,
                "booster": bool(use_booster and self.booster is not None), "selected": 0}

        best = 0
        if n_cand > 1:
            previews = self._generate(cond1, z, self.search_steps, gen_batch_size, decode_batch_size)
            clap = self.clap.score(previews, self.sample_rate, prompt, chunk=clap_chunk)
            aes = self.aesthetics.score(previews, self.sample_rate, chunk=clap_chunk)
            values = {"clap": clap.tolist(), **{a: aes[a].tolist() for a in AES_AXES}}
            best = select_best(values, n_cand, rule)
            info.update(scores=values, selected=best, rule=rule)

        lat = self.generator.sample(cond1, z[best:best + 1], k_steps, self.omega)
        wav = self.vae.decode(lat.transpose(2, 1)).sample[0].cpu()
        wav = wav[:, : min(wav.shape[-1], int(round(duration * self.sample_rate)))]
        return (wav, info) if return_info else wav
