"""Replay a request trace through a RadixCache and collect metrics.

Time-to-first-token is estimated from computed (non-cached) prompt tokens using
a prefill throughput you can calibrate on your own machine with
``engine/model.py`` (``python -m engine.model --calibrate``).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from engine.radix_cache import RadixCache, Record
from workloads.trace import Request


@dataclass
class AgentStats:
    requests: int = 0
    prompt_tokens: int = 0
    hit_tokens: int = 0
    computed_tokens: int = 0
    ttft_ms: List[float] = field(default_factory=list)
    # Tokens from non-request traffic (keep-alive pings): billed, but not in hit rate.
    overhead_sent: int = 0
    overhead_hit: int = 0
    overhead_computed: int = 0

    @property
    def total_computed(self) -> int:
        return self.computed_tokens + self.overhead_computed

    @property
    def hit_rate(self) -> float:
        return self.hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class RunResult:
    per_agent: Dict[str, AgentStats]
    records: List[Record]
    evictions: int
    capacity: int

    @property
    def prompt_tokens(self) -> int:
        return sum(a.prompt_tokens for a in self.per_agent.values())

    @property
    def hit_tokens(self) -> int:
        return sum(a.hit_tokens for a in self.per_agent.values())

    @property
    def computed_tokens(self) -> int:
        return sum(a.computed_tokens for a in self.per_agent.values())

    @property
    def total_computed(self) -> int:
        """All prefill compute, including keep-alive traffic (what billing splits)."""
        return sum(a.total_computed for a in self.per_agent.values())

    @property
    def hit_rate(self) -> float:
        return self.hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def ttft_percentile(self, q: float) -> float:
        xs = sorted(t for a in self.per_agent.values() for t in a.ttft_ms)
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
        return xs[idx]


def simulate(
    trace: Iterable[Request],
    capacity_tokens: int,
    eviction,
    prefill_tps: float = 2000.0,
    overhead_ms: float = 15.0,
    block_size: int = 16,
    count_agents: Optional[set] = None,
) -> RunResult:
    """Run ``trace`` and return per-agent stats plus usage records.

    ``count_agents``: if given, only these agents' requests are recorded.
    Keep-alive pings are recorded as overhead (billed, but not in hit rate).
    """
    cache = RadixCache(capacity_tokens, eviction, block_size=block_size)
    stats: Dict[str, AgentStats] = defaultdict(AgentStats)
    for req in trace:
        res = cache.request(req.agent, req.tokens, req.time)
        if count_agents is not None and req.agent not in count_agents:
            continue
        s = stats[req.agent]
        if req.kind != "request":
            s.overhead_sent += res.prompt_tokens
            s.overhead_hit += res.hit_tokens
            s.overhead_computed += res.computed_tokens
            continue
        s.requests += 1
        s.prompt_tokens += res.prompt_tokens
        s.hit_tokens += res.hit_tokens
        s.computed_tokens += res.computed_tokens
        s.ttft_ms.append(overhead_ms + 1000.0 * res.computed_tokens / prefill_tps)
    return RunResult(dict(stats), cache.all_records(), cache.evictions, capacity_tokens)


def jain_index(xs: List[float]) -> float:
    """Jain's fairness index: 1 = perfectly equal, 1/n = one agent gets all."""
    xs = [x for x in xs if x is not None]
    if not xs or sum(xs) == 0:
        return 1.0
    return sum(xs) ** 2 / (len(xs) * sum(x * x for x in xs))


def fairness(run: RunResult, ideal: RunResult) -> Dict[str, float]:
    """Per-agent slowdown = computed tokens vs. an infinite-cache run.

    Returns Jain's index over 1/slowdown and the max/min slowdown ratio.
    """
    slow = {}
    for a, s in run.per_agent.items():
        base = max(1, ideal.per_agent[a].computed_tokens)
        slow[a] = s.computed_tokens / base
    inv = [1.0 / v for v in slow.values() if v > 0]
    return {
        "jain": jain_index(inv),
        "max_min_slowdown": max(slow.values()) / max(1e-9, min(slow.values())),
        "slowdown": slow,
    }
