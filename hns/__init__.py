"""Hybrid Noise Skimming (HNS) for full-band text-to-audio generation."""

from .pipeline import HNSPipeline
from .search import budget, select_best
from .xmf import XMFGenerator
from .noise_booster import NoiseBooster

__all__ = ["HNSPipeline", "XMFGenerator", "NoiseBooster", "budget", "select_best"]
__version__ = "1.0.0"
