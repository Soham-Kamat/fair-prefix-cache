"""Synthetic multi-agent workloads.

Token lengths are rough but realistic orders of magnitude for agent prompts.
Each generator is seeded, so traces are reproducible.
"""
from __future__ import annotations

import random
from typing import List

from workloads.trace import Request, SegmentStore


def coding_team(steps: int = 60, seed: int = 0, store: SegmentStore | None = None) -> List[Request]:
    """Planner, coder and reviewer share a system prompt, tool specs and repo
    context; each also has its own role prompt and growing history."""
    rng = random.Random(seed)
    st = store or SegmentStore()
    agents = ["planner", "coder", "reviewer"]
    shared = [("team/system", 2000), ("team/tools", 1500), ("team/repo", 3000)]
    history = {a: [] for a in agents}
    trace, t = [], 0.0
    for step in range(steps):
        a = rng.choice(agents)
        msg = (f"{a}/turn{step}", rng.randint(100, 400))
        prompt = shared + [(f"{a}/role", 300)] + history[a] + [msg]
        trace.append(Request(t, a, st.prompt(prompt)))
        history[a].append(msg)
        history[a].append((f"{a}/reply{step}", rng.randint(100, 300)))
        # Long histories get summarised: keep only the last 8 turns.
        history[a] = history[a][-16:]
        t += rng.expovariate(1.0)
    return trace


def rag(n_agents: int = 8, n_docs: int = 40, steps: int = 400, zipf_s: float = 1.1,
        seed: int = 0, store: SegmentStore | None = None) -> List[Request]:
    """Agents answer questions over a shared document set with Zipf-popular docs."""
    rng = random.Random(seed)
    st = store or SegmentStore()
    weights = [1.0 / (i + 1) ** zipf_s for i in range(n_docs)]
    agents = [f"rag{i}" for i in range(n_agents)]
    trace, t = [], 0.0
    for step in range(steps):
        a = rng.choice(agents)
        d1, d2 = rng.choices(range(n_docs), weights=weights, k=2)
        prompt = [("rag/system", 500), (f"{a}/prompt", 200),
                  (f"doc{d1}", 1000), (f"doc{d2}", 1000), (f"{a}/q{step}", 100)]
        trace.append(Request(t, a, st.prompt(prompt)))
        t += rng.expovariate(1.0)
    return trace


def multi_tenant_chat(n_tenants: int = 10, users_per_tenant: int = 3, steps: int = 500,
                      heavy_tenants: int = 2, seed: int = 0,
                      store: SegmentStore | None = None) -> List[Request]:
    """Tenants with their own system prompts; a few heavy tenants send most
    traffic. The agent id is the tenant (the billing unit)."""
    rng = random.Random(seed)
    st = store or SegmentStore()
    tenants = [f"tenant{i}" for i in range(n_tenants)]
    w = [5.0 if i < heavy_tenants else 1.0 for i in range(n_tenants)]
    history = {}
    trace, t = [], 0.0
    for step in range(steps):
        ten = rng.choices(tenants, weights=w)[0]
        user = f"{ten}/u{rng.randrange(users_per_tenant)}"
        h = history.setdefault(user, [])
        msg = (f"{user}/m{step}", rng.randint(50, 200))
        prompt = [(f"{ten}/system", 1000)] + h + [msg]
        trace.append(Request(t, ten, st.prompt(prompt)))
        h.append(msg)
        h.append((f"{user}/r{step}", rng.randint(50, 250)))
        history[user] = h[-12:]
        t += rng.expovariate(1.0)
    return trace


WORKLOADS = {
    "coding": coding_team,
    "rag": rag,
    "chat": multi_tenant_chat,
}
