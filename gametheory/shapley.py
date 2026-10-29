"""Shapley values for cache cost-sharing games.

The cache game: players are agents, and C(S) is the compute needed if only the
agents in S had used the cache. Approximating the cache history as fixed, a
record (one computed run of tokens) is needed by S iff some member of S used
it, so

    C(S) = sum over records r with users(r) ∩ S ≠ ∅ of cost(r).

This is a sum of "airport/union" games, one per record, and the Shapley value
is linear, so phi_i = sum over records used by i of cost(r) / |users(r)|.
``exact`` brute-forces the definition to verify that closed form; ``monte_carlo``
samples permutations for games without a closed form (e.g. once eviction
depends on who is present).
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Callable, Dict, FrozenSet, Iterable, List, Sequence

CostFn = Callable[[FrozenSet[str]], float]


def union_cost_fn(records: Iterable) -> CostFn:
    recs = [(frozenset(r.users) or frozenset([r.writer]), r.cost) for r in records]

    def C(S: FrozenSet[str]) -> float:
        return float(sum(c for users, c in recs if users & S))

    return C


def exact(players: Sequence[str], C: CostFn) -> Dict[str, float]:
    """Shapley value by the subset formula. O(2^n) - use for n <= ~15."""
    n = len(players)
    phi = {p: 0.0 for p in players}
    fact = [math.factorial(k) for k in range(n + 1)]
    cache: Dict[FrozenSet[str], float] = {}

    def cost(S):
        if S not in cache:
            cache[S] = C(S)
        return cache[S]

    for i in players:
        others = [p for p in players if p != i]
        for k in range(n):
            w = fact[k] * fact[n - k - 1] / fact[n]
            for S in itertools.combinations(others, k):
                S = frozenset(S)
                phi[i] += w * (cost(S | {i}) - cost(S))
    return phi


def closed_form(records: Iterable, players: Sequence[str]) -> Dict[str, float]:
    phi = {p: 0.0 for p in players}
    for r in records:
        users = [u for u in r.users if u in phi] or [r.writer]
        for u in users:
            phi[u] += r.cost / len(users)
    return phi


def monte_carlo(players: Sequence[str], C: CostFn, samples: int = 2000, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    phi = {p: 0.0 for p in players}
    order: List[str] = list(players)
    for _ in range(samples):
        rng.shuffle(order)
        S: FrozenSet[str] = frozenset()
        prev = C(S)
        for p in order:
            S = S | {p}
            cur = C(S)
            phi[p] += cur - prev
            prev = cur
    return {p: v / samples for p, v in phi.items()}
