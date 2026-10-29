"""Billing rules: split the total compute cost of a run among agents.

All rules except ``provider_cached`` are *budget balanced*: shares add up to
the total tokens actually computed. ``provider_cached`` mimics API pricing
where a cache write costs more than base input and a cache read much less;
compare it by normalised shares.

* ``per_token_sent``     - proportional to prompt tokens each agent sent.
* ``equal_split``        - total / number of agents.
* ``pay_what_you_compute`` - each agent pays for the runs its own requests
                           computed (first writer pays, later readers free-ride).
* ``shapley``            - each computed run's cost split equally among the
                           agents that used it while cached. This is the exact
                           Shapley value of the union-cost game (see
                           ``gametheory/shapley.py`` and its tests).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from engine.radix_cache import Record
from engine.simulator import RunResult


def per_token_sent(run: RunResult) -> Dict[str, float]:
    total = run.total_computed
    sent = {a: s.prompt_tokens + s.overhead_sent for a, s in run.per_agent.items()}
    z = sum(sent.values()) or 1
    return {a: total * v / z for a, v in sent.items()}


def equal_split(run: RunResult) -> Dict[str, float]:
    n = len(run.per_agent) or 1
    return {a: run.total_computed / n for a in run.per_agent}


def pay_what_you_compute(run: RunResult) -> Dict[str, float]:
    return {a: float(s.total_computed) for a, s in run.per_agent.items()}


def shapley_from_records(records: List[Record], agents) -> Dict[str, float]:
    bill: Dict[str, float] = defaultdict(float)
    for rec in records:
        users = [u for u in rec.users if u in agents] or [rec.writer]
        share = rec.cost / len(users)
        for u in users:
            bill[u] += share
    return {a: bill.get(a, 0.0) for a in agents}


def shapley(run: RunResult) -> Dict[str, float]:
    return shapley_from_records(run.records, set(run.per_agent))


def provider_cached(run: RunResult, write_mult: float = 1.25, read_mult: float = 0.1) -> Dict[str, float]:
    """Price-list style bill in base-input-token units (multipliers are
    illustrative; check your provider's current pricing)."""
    return {a: write_mult * s.total_computed + read_mult * (s.hit_tokens + s.overhead_hit)
            for a, s in run.per_agent.items()}


RULES = {
    "per_token_sent": per_token_sent,
    "equal_split": equal_split,
    "pay_what_you_compute": pay_what_you_compute,
    "shapley": shapley,
    "provider_cached": provider_cached,
}


def normalise(bill: Dict[str, float]) -> Dict[str, float]:
    z = sum(bill.values()) or 1.0
    return {a: v / z for a, v in bill.items()}


def billing_error(bill: Dict[str, float], reference: Dict[str, float]) -> float:
    """Total-variation distance between normalised shares (0 = identical, 1 = disjoint)."""
    p, q = normalise(bill), normalise(reference)
    return 0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in set(p) | set(q))
