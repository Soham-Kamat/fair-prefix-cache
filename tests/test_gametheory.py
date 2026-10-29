import random

import pytest

from engine.radix_cache import Record
from engine.simulator import simulate
from gametheory import core_check, shapley
from gametheory import strategic_agents as sa
from policies import billing, eviction
from workloads.agents import coding_team


def random_records(n_players=5, n_records=12, seed=0):
    rng = random.Random(seed)
    players = [f"p{i}" for i in range(n_players)]
    recs = []
    for _ in range(n_records):
        users = rng.sample(players, rng.randint(1, n_players))
        recs.append(Record(cost=rng.randint(1, 100), writer=users[0], users={u: 1 for u in users}))
    return players, recs


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_equals_exact_shapley(seed):
    players, recs = random_records(seed=seed)
    C = shapley.union_cost_fn(recs)
    ex = shapley.exact(players, C)
    cf = shapley.closed_form(recs, players)
    for p in players:
        assert ex[p] == pytest.approx(cf[p])


def test_monte_carlo_close_to_exact():
    players, recs = random_records(seed=7)
    C = shapley.union_cost_fn(recs)
    ex = shapley.exact(players, C)
    mc = shapley.monte_carlo(players, C, samples=4000)
    total = sum(ex.values())
    for p in players:
        assert abs(mc[p] - ex[p]) < 0.05 * total


@pytest.mark.parametrize("seed", range(5))
def test_shapley_allocation_in_core(seed):
    players, recs = random_records(seed=seed)
    C = shapley.union_cost_fn(recs)
    assert core_check.in_core(shapley.closed_form(recs, players), players, C)


def test_writer_pays_differs_from_shapley():
    # p0 writes a 10-token run both use; under writer-pays p1 free-rides entirely.
    recs = [Record(cost=10, writer="p0", users={"p0": 1, "p1": 1}), Record(cost=1, writer="p0", users={"p0": 1})]
    players = ["p0", "p1"]
    C = shapley.union_cost_fn(recs)
    writer_pays = {"p0": 11.0, "p1": 0.0}
    assert core_check.in_core(writer_pays, players, C)          # stable, but unfair:
    sh = shapley.closed_form(recs, players)
    assert sh == {"p0": 6.0, "p1": 5.0}                         # Shapley shares the run


def test_billing_rules_budget_balanced():
    run = simulate(coding_team(steps=40), 15000, eviction.LRU())
    for name in ("per_token_sent", "equal_split", "pay_what_you_compute", "shapley"):
        assert sum(billing.RULES[name](run).values()) == pytest.approx(run.total_computed)


def test_late_arrival_free_rides_only_without_shapley():
    trace = coding_team(steps=60)
    g = sa.cheater_gain(trace, lambda tr: sa.late_arrival(tr, "coder", 0.5), "coder", 30000, eviction.LRU)
    assert g["bill_saving_pct[pay_what_you_compute]"] > 5
    assert abs(g["bill_saving_pct[shapley]"]) < 1e-9
