#!/usr/bin/env python
"""Generate audio from text with Hybrid Noise Skimming.

    # one prompt
    python generate.py --prompt "A dog barks while birds chirp" --nfe 16 --out dog.wav

    # one prompt per line in a text file
    python generate.py --prompt_file prompts.txt --nfe 8 --out_dir outputs/

The NFE budget is split by the paper's rule: NFE <= 3 uses the Noise Booster
and NFE-step generation; NFE > 3 runs the Single-Step Search over N = NFE - 2
boosted candidates and generates the winner in 2 steps.
"""

import argparse
import json
import os
import re
import time

import torch
import torchaudio

from hns import HNSPipeline
from hns.pipeline import DEFAULT_VAE
from hns.search import SELECTION_RULES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="text prompt")
    src.add_argument("--prompt_file", type=str, help="text file, one prompt per line")
    p.add_argument("--nfe", type=int, default=16,
                   help="total generator budget (1, 2, 4, 8, 16, 32, 64, ...)")
    p.add_argument("--duration", type=float, default=10.0, help="seconds, at most 10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="output wav (single prompt)")
    p.add_argument("--out_dir", type=str, default="outputs", help="output folder (prompt file)")
    p.add_argument("--checkpoints", type=str, default="checkpoints",
                   help="folder holding xmf_generator.safetensors, noise_booster.safetensors "
                        "and the LAION-CLAP checkpoint; missing files are downloaded into it "
                        "from the Hugging Face hub on first use")
    p.add_argument("--vae", type=str, default=DEFAULT_VAE,
                   help="Stable Audio Open VAE: hub id (default) or a local directory")
    p.add_argument("--clap_ckpt", type=str, default=None)
    p.add_argument("--aes_ckpt", type=str, default=None)
    p.add_argument("--no_booster", action="store_true", help="plain x-MF, without the Noise Booster")
    p.add_argument("--rule", choices=list(SELECTION_RULES), default="rank_mean",
                   help="verifier rule of the Single-Step Search")
    p.add_argument("--num_candidates", type=int, default=None,
                   help="override N (default: from --nfe)")
    p.add_argument("--num_steps", type=int, default=None,
                   help="override K, the final generation steps (default: from --nfe)")
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="candidates per generator forward during the search (0 = all)")
    p.add_argument("--decode_batch_size", type=int, default=2,
                   help="candidates per VAE decode during the search (0 = all)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def slug(text: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s[:n] or "prompt"


def main():
    args = parse_args()
    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    pipe = HNSPipeline.from_pretrained(
        checkpoint_dir=args.checkpoints, vae=args.vae, device=args.device,
        use_booster=not args.no_booster, clap_ckpt=args.clap_ckpt, aes_ckpt=args.aes_ckpt,
        use_verifiers=(args.num_candidates or args.nfe) > 1,
    )

    if args.prompt is not None and args.out is not None:
        out_paths = [args.out]
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        out_paths = [os.path.join(args.out_dir, f"{i:03d}_{slug(pr)}.wav")
                     for i, pr in enumerate(prompts)]

    for prompt, out_path in zip(prompts, out_paths):
        t0 = time.time()
        wav, info = pipe.generate(
            prompt, nfe=args.nfe, duration=args.duration, seed=args.seed,
            num_candidates=args.num_candidates, num_steps=args.num_steps,
            use_booster=not args.no_booster, rule=args.rule,
            gen_batch_size=args.gen_batch_size, decode_batch_size=args.decode_batch_size,
            return_info=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torchaudio.save(out_path, wav, pipe.sample_rate)
        msg = (f"[{time.time() - t0:5.2f}s] NFE {info['nfe']:3d} = {info['num_candidates']} candidate(s) "
               f"+ {info['final_steps']}-step generation -> {out_path}")
        if "scores" in info:
            i = info["selected"]
            msg += (f"  (selected #{i}: CLAP {info['scores']['clap'][i]:.3f}, "
                    f"PQ {info['scores']['PQ'][i]:.2f})")
        print(msg, flush=True)
        if len(prompts) > 1 and "scores" in info:
            with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
                json.dump({"prompt": prompt, **info}, f, indent=1)


if __name__ == "__main__":
    main()
