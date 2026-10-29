"""Eviction policies. Each picks one leaf to evict from ``candidates``.

* ``LRU``    - least recently used leaf (the usual default).
* ``LFU``    - fewest hits, ties broken by recency.
* ``Quota``  - each agent owns what it *wrote*; evict from the agent furthest
               over an equal share of memory, LRU within that agent.
* ``ShapleyShare`` - memory occupancy is charged by Shapley share: a cached run
               of L tokens used by k agents charges L/k to each. Evict from the
               agent with the largest charge, choosing among leaves it uses
               the one with the fewest co-users (LRU tie-break), so shared
               runs survive longest. Sharing is cheap for everyone, and
               keep-alive pings on a private prefix do not lower its charge.

TODO (good extension): decay ``record.users`` over a time window so agents
that stopped using a run stop sharing its charge.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class LRU:
    name = "lru"

    def choose_victim(self, cache, candidates, now):
        return min(candidates, key=lambda n: n.last_access)


class LFU:
    name = "lfu"

    def choose_victim(self, cache, candidates, now):
        return min(candidates, key=lambda n: (n.hits, n.last_access))


class Quota:
    """Writer-owns-it quota: the agent that computed a run is charged for all of it."""

    name = "quota"

    def occupancy(self, cache) -> Dict[str, float]:
        occ: Dict[str, float] = defaultdict(float)
        for nd in cache.nodes():
            if nd.record:
                occ[nd.record.writer] += len(nd)
        return occ

    def choose_victim(self, cache, candidates, now):
        occ = self.occupancy(cache)
        owners = {nd.record.writer for nd in candidates if nd.record}
        if not owners:
            return min(candidates, key=lambda n: n.last_access)
        worst = max(owners, key=lambda a: occ.get(a, 0.0))
        mine = [nd for nd in candidates if nd.record and nd.record.writer == worst]
        return min(mine, key=lambda n: n.last_access)


class ShapleyShare:
    """Shapley-share occupancy with an equal fair share per active agent."""

    name = "shapley"

    def occupancy(self, cache) -> Dict[str, float]:
        occ: Dict[str, float] = defaultdict(float)
        for nd in cache.nodes():
            if nd.record and nd.record.users:
                share = len(nd) / len(nd.record.users)
                for a in nd.record.users:
                    occ[a] += share
        return occ

    def choose_victim(self, cache, candidates, now):
        occ = self.occupancy(cache)
        users_of = {id(nd): (nd.record.users if nd.record else {}) for nd in candidates}
        agents = {a for nd in candidates for a in users_of[id(nd)]}
        if not agents:
            return min(candidates, key=lambda n: n.last_access)
        worst = max(agents, key=lambda a: occ.get(a, 0.0))
        mine = [nd for nd in candidates if worst in users_of[id(nd)]]
        # Prefer leaves the worst agent uses alone (fewest co-users), then LRU.
        return min(mine, key=lambda n: (len(users_of[id(n)]), n.last_access))


POLICIES = {p.name: p for p in (LRU, LFU, Quota, ShapleyShare)}


def make(name: str):
    return POLICIES[name]()
