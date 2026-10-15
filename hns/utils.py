"""Small helpers: configs, checkpoints, deterministic noise."""

import os
from typing import Dict, Optional

import torch
import yaml

__all__ = ["REPO_ROOT", "HF_REPO", "load_yaml", "load_safetensors", "resolve_checkpoint",
           "gaussian_noise"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## Hugging Face repository holding the released x-MF generator and Noise Booster.
HF_REPO = "briskmuntjack/hybrid-noise-skimming"


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_checkpoint(checkpoint_dir: str, filename: str, repo_id: Optional[str] = HF_REPO) -> str:
    """Path of `<checkpoint_dir>/<filename>`, downloading it from `repo_id` on
    the Hugging Face hub into `checkpoint_dir` when it is not there yet.
    `repo_id=None` disables the download and only checks the local file."""
    path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(path):
        return path
    if repo_id is None:
        raise FileNotFoundError(f"checkpoint not found: {path}")
    from huggingface_hub import hf_hub_download

    print(f"[checkpoints] {filename} not found in {checkpoint_dir}; downloading from {repo_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return load_file(path)


def gaussian_noise(num: int, seq_len: int, channels: int, seed: int = 0) -> torch.Tensor:
    """num i.i.d. standard-normal noises (num, 1, seq_len, channels), drawn on
    the CPU so the same seed gives the same noise on any device."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(num, 1, seq_len, channels, generator=g)
