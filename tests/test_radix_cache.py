import random

from engine.radix_cache import RadixCache
from policies import eviction


def test_prefix_reuse_and_split():
    c = RadixCache(1000, eviction.LRU())
    r1 = c.request("a", [1, 2, 3, 4, 5], now=0)
    assert (r1.hit_tokens, r1.computed_tokens) == (0, 5)
    r2 = c.request("b", [1, 2, 3, 9, 9], now=1)   # shares 3 tokens, splits the edge
    assert (r2.hit_tokens, r2.computed_tokens) == (3, 2)
    r3 = c.request("a", [1, 2, 3, 4, 5], now=2)
    assert r3.hit_tokens == 5
    c.check_invariants()
    assert c.size == 7


def test_split_records_keep_costs_and_users():
    c = RadixCache(1000, eviction.LRU())
    c.request("a", list(range(10)), now=0)
    c.request("b", list(range(4)) + [99], now=1)
    recs = c.all_records()
    assert sum(r.cost for r in recs) == 11            # 10 computed by a, 1 by b
    shared = [r for r in recs if r.cost == 4][0]
    assert set(shared.users) == {"a", "b"}


def test_capacity_respected_under_random_load():
    rng = random.Random(0)
    for pol in eviction.POLICIES:
        c = RadixCache(300, eviction.make(pol))
        for t in range(400):
            base = rng.choice([[1] * 0, list(range(100, 150)), list(range(200, 260))])
            toks = base + [rng.randrange(1000, 1100) for _ in range(rng.randint(1, 60))]
            c.request(rng.choice("abc"), toks, now=t)
            c.check_invariants()
        assert c.size <= 300


def test_compute_is_conserved():
    """Every computed token appears in exactly one record (evicted or live)."""
    rng = random.Random(1)
    c = RadixCache(200, eviction.LRU())
    computed = 0
    for t in range(300):
        toks = list(range(rng.choice([0, 50]), 50 + rng.randint(1, 80)))
        computed += c.request(rng.choice("xyz"), toks, now=t).computed_tokens
    assert sum(r.cost for r in c.all_records()) == computed


def test_oversized_request_not_cached():
    c = RadixCache(10, eviction.LRU())
    r = c.request("a", list(range(50)), now=0)
    assert not r.cached and c.size == 0
