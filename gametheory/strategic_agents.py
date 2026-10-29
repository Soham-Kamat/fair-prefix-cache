"""Strategic (cheating) agents, as trace transformations.

* ``keepalive``    - the cheater pings its own prefix every ``every`` time
                     units to keep it hot. Under LRU this protects its cache at
                     others' expense; the pings are cheap cache reads.
* ``late_arrival`` - the cheater delays its requests so others compute shared
                     prefixes first. Under pay-what-you-compute billing the
                     first writer pays everything, so arriving late free-rides.

``cheater_gain`` reruns a workload honest vs. cheating and reports what the
cheater gained (in bill and in hit rate) and what everyone else lost.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from engine.simulator import simulate
from policies import billing
from workloads.trace import Request


def keepalive(trace: List[Request], agent: str, prefix_len: Optional[int] = None, every: float = 2.0) -> List[Request]:
    """Ping the first ``prefix_len`` tokens of the agent's first prompt
    (default: the whole first prompt) every ``every`` time units."""
    mine = [r for r in trace if r.agent == agent]
    if not mine:
        return list(trace)
    prefix = mine[0].tokens[:prefix_len] if prefix_len else mine[0].tokens
    t_end = trace[-1].time
    pings = []
    t = mine[0].time + every
    while t < t_end:
        pings.append(Request(t, agent, prefix, kind="keepalive"))
        t += every
    return sorted(trace + pings, key=lambda r: r.time)


def late_arrival(trace: List[Request], agent: str, delay: float = 0.5) -> List[Request]:
    out = [Request(r.time + delay, r.agent, r.tokens, r.kind) if r.agent == agent else r for r in trace]
    return sorted(out, key=lambda r: r.time)


def cheater_gain(
    trace: List[Request],
    cheat: Callable[[List[Request]], List[Request]],
    cheater: str,
    capacity: int,
    make_policy: Callable[[], object],
    rules: Sequence[str] = ("per_token_sent", "pay_what_you_compute", "shapley", "provider_cached"),
) -> Dict[str, float]:
    honest = simulate(trace, capacity, make_policy())
    cheating = simulate(cheat(trace), capacity, make_policy())
    out: Dict[str, float] = {}
    h_hr, c_hr = honest.per_agent[cheater].hit_rate, cheating.per_agent[cheater].hit_rate
    out["cheater_hit_rate_gain_pp"] = 100 * (c_hr - h_hr)
    others = [a for a in honest.per_agent if a != cheater]
    oh = sum(honest.per_agent[a].hit_tokens for a in others) / max(1, sum(honest.per_agent[a].prompt_tokens for a in others))
    oc = sum(cheating.per_agent[a].hit_tokens for a in others) / max(1, sum(cheating.per_agent[a].prompt_tokens for a in others))
    out["others_hit_rate_change_pp"] = 100 * (oc - oh)
    for rule in rules:
        hb = billing.RULES[rule](honest)
        cb = billing.RULES[rule](cheating)
        h = hb.get(cheater, 0.0)
        c = cb.get(cheater, 0.0)
        # % reduction in the cheater's bill (positive = cheating paid off)
        out[f"bill_saving_pct[{rule}]"] = 100 * (h - c) / h if h else 0.0
    return out
