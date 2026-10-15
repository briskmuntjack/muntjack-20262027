"""Single-Step Search (SSS): candidate selection from one-step previews.

Given verifier scores for N candidates of the same prompt, the rank-mean rule
(Eq. "rankmean" in the paper) picks

    i* = argmax_i 1/2 ( rk(s^clap)_i + rk(s^pq)_i )

where rk() is the ascending within-pool rank (best candidate = rank N) and
exact ties share the mean of the positions they occupy. Ties in the final
argmax resolve to the lowest index. Rank aggregation needs no calibration
between the CLAP and PQ scales.

`budget(nfe)` is the budget rule of the paper (Eq. "budget"):

    nfe <= 3 : N = 1,       K = nfe   (Noise Booster + K-step generation)
    nfe >  3 : N = nfe - 2, K = 2     (Noise Booster + SSS + 2-step generation)
"""

from typing import Dict, List, Sequence, Tuple

__all__ = ["SELECTION_RULES", "budget", "rank_mean_scores", "selection_scores", "select_best"]

## selection rule -> verifier axes it reads
SELECTION_RULES = {
    "rank_mean": ("clap", "PQ"),  # paper default
    "clap": ("clap",),
    "pq": ("PQ",),
}


def budget(nfe: int, final_steps: int = 2) -> Tuple[int, int]:
    """Total generator budget -> (num_candidates N, final steps K)."""
    if nfe < 1:
        raise ValueError("nfe must be >= 1")
    if nfe <= final_steps + 1:
        return 1, nfe
    return nfe - final_steps, final_steps


def rank_mean_scores(values_by_axis: Dict[str, Sequence[float]]) -> List[float]:
    """-(mean rank over axes) per candidate; higher is better.

    Rank 1 = best on an axis; ties share the mean of the positions they span.
    """
    n = len(next(iter(values_by_axis.values())))
    totals = [0.0] * n
    for vals in values_by_axis.values():
        order = sorted(range(n), key=lambda i: -vals[i])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                totals[order[k]] += shared
            i = j + 1
    return [-t / len(values_by_axis) for t in totals]


def selection_scores(values: Dict[str, Sequence[float]], rule: str = "rank_mean") -> List[float]:
    """One selection score per candidate (higher is better) under `rule`."""
    if rule not in SELECTION_RULES:
        raise ValueError(f"unknown selection rule '{rule}'; choose from {list(SELECTION_RULES)}")
    axes = SELECTION_RULES[rule]
    if rule == "rank_mean":
        return rank_mean_scores({a: [float(v) for v in values[a]] for a in axes})
    return [float(v) for v in values[axes[0]]]


def select_best(values: Dict[str, Sequence[float]], n: int = None, rule: str = "rank_mean") -> int:
    """Index of the winner among the FIRST n candidates (ranks are computed
    within that prefix, which is what best-of-n means)."""
    axes = SELECTION_RULES[rule]
    n = n or len(values[axes[0]])
    prefix = {a: list(values[a])[:n] for a in axes}
    scores = selection_scores(prefix, rule)
    return max(range(n), key=lambda c: scores[c])
