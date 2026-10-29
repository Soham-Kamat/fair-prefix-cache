"""Run every eviction policy x cache size x workload, plus billing and cheater
experiments. Writes CSVs to ``results/``.

    python -m experiments.run_sweep                # all workloads
    python -m experiments.run_sweep --workloads rag --sizes 0.05 0.1 0.2
    python -m experiments.run_sweep --prefill-tps 1800   # from engine.model --calibrate
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

from engine.simulator import fairness, simulate
from gametheory import strategic_agents as sa
from policies import billing, eviction
from workloads.agents import WORKLOADS

POLICY_NAMES = ["lru", "lfu", "quota", "shapley"]


def unique_tokens(trace) -> int:
    """Tokens an infinite cache would hold (sets the scale for cache sizes)."""
    run = simulate(trace, 10**12, eviction.LRU())
    return sum(r.cost for r in run.records)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--sizes", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4],
                    help="cache capacity as a fraction of the workload's unique tokens")
    ap.add_argument("--prefill-tps", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    sweep, bills, cheats = [], [], []
    for wl in args.workloads:
        trace = WORKLOADS[wl](seed=args.seed)
        uniq = unique_tokens(trace)
        ideal = simulate(trace, 10**12, eviction.LRU(), prefill_tps=args.prefill_tps)
        for frac in args.sizes:
            cap = max(1, int(frac * uniq))
            for pol in POLICY_NAMES:
                run = simulate(trace, cap, eviction.make(pol), prefill_tps=args.prefill_tps)
                fair = fairness(run, ideal)
                sweep.append({
                    "workload": wl, "cache_frac": frac, "capacity_tokens": cap, "policy": pol,
                    "hit_rate": round(run.hit_rate, 4),
                    "recomputed_tokens_per_req": round(run.computed_tokens / max(1, sum(s.requests for s in run.per_agent.values())), 1),
                    "ttft_p50_ms": round(run.ttft_percentile(50), 1),
                    "ttft_p99_ms": round(run.ttft_percentile(99), 1),
                    "jain": round(fair["jain"], 4),
                    "max_min_slowdown": round(fair["max_min_slowdown"], 3),
                    "evictions": run.evictions,
                })
                ref = billing.shapley(run)
                row = {"workload": wl, "cache_frac": frac, "policy": pol}
                for rule in billing.RULES:
                    row[f"err_vs_shapley[{rule}]"] = round(billing.billing_error(billing.RULES[rule](run), ref), 4)
                bills.append(row)
            print(f"{wl:7s} cache={frac:<5} done")

        # Cheater experiments at a tight cache (the contested regime).
        cap = max(1, int(min(args.sizes) * 2 * uniq))
        counts: Dict[str, int] = {}
        for r in trace:
            counts[r.agent] = counts.get(r.agent, 0) + 1
        cheater = min(sorted(counts), key=counts.get)   # a light user gains most from cheating
        attacks = {
            "keepalive": lambda tr: sa.keepalive(tr, cheater, every=2.0),
            "late_arrival": lambda tr: sa.late_arrival(tr, cheater, delay=0.5),
        }
        for attack, fn in attacks.items():
            for pol in POLICY_NAMES:
                g = sa.cheater_gain(trace, fn, cheater, cap, lambda p=pol: eviction.make(p))
                cheats.append({"workload": wl, "attack": attack, "policy": pol, "cheater": cheater,
                               **{k: round(v, 2) for k, v in g.items()}})

    write_csv(os.path.join(args.out, "sweep.csv"), sweep)
    write_csv(os.path.join(args.out, "billing.csv"), bills)
    write_csv(os.path.join(args.out, "cheaters.csv"), cheats)
    print(f"wrote {args.out}/sweep.csv, billing.csv, cheaters.csv")


if __name__ == "__main__":
    main()
