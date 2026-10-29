"""Is a cost allocation in the core?

An allocation x is in the core of a cost game if it is efficient
(sum x = C(N)) and no coalition pays more than it would alone:
sum_{i in S} x_i <= C(S) for every S. Union-cost games are concave
(submodular), so the Shapley value is always in their core - a good result
to state and verify empirically.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

from gametheory.shapley import CostFn


def core_violations(x: Dict[str, float], players: Sequence[str], C: CostFn, tol: float = 1e-6) -> List[Tuple[frozenset, float]]:
    """Coalitions that would rather leave, with the amount they overpay."""
    out = []
    total = C(frozenset(players))
    if abs(sum(x.values()) - total) > tol * max(1.0, total):
        out.append((frozenset(players), sum(x.values()) - total))
    for k in range(1, len(players)):
        for S in itertools.combinations(players, k):
            S = frozenset(S)
            over = sum(x[i] for i in S) - C(S)
            if over > tol * max(1.0, C(S)):
                out.append((S, over))
    return out


def in_core(x, players, C) -> bool:
    return not core_violations(x, players, C)
