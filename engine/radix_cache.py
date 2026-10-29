"""Radix-tree prefix cache with per-node usage records.

Each node owns a run of tokens (an edge of the radix tree) whose KV cache is
held in memory. Memory is counted in tokens; ``block_size`` only converts that
to KV blocks for reporting.

Every cached node carries a *record*: the compute cost paid to create it
(its token count) and the set of agents that used it while it was cached.
Records are the input to Shapley billing (see ``policies/billing``): in the
union-cost game each record's cost is split equally among its users.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Record:
    """One computation of a token run, and who benefited from it."""

    cost: int                      # tokens computed (prefill cost)
    writer: str                    # agent whose request computed it
    users: Dict[str, int] = field(default_factory=dict)  # agent -> times used

    def split(self, k: int, total: int) -> "Record":
        """Split off the cost of the last ``total - k`` tokens into a new record."""
        tail = Record(cost=total - k, writer=self.writer, users=dict(self.users))
        self.cost = k
        return tail


class Node:
    __slots__ = ("tokens", "children", "parent", "last_access", "hits", "record", "lock")

    def __init__(self, tokens: Tuple[int, ...], parent: Optional["Node"], record: Optional[Record], now: float):
        self.tokens = tokens
        self.children: Dict[int, Node] = {}
        self.parent = parent
        self.last_access = now
        self.hits = 0
        self.record = record
        self.lock = 0              # >0 while on the path of the request being served

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node(len={len(self)}, hits={self.hits}, users={list(self.record.users) if self.record else []})"


@dataclass
class RequestResult:
    agent: str
    prompt_tokens: int
    hit_tokens: int
    computed_tokens: int
    cached: bool                   # False if the new suffix did not fit in memory


class RadixCache:
    """Prefix cache over token sequences with pluggable eviction.

    ``eviction`` is any object with ``choose_victim(cache, candidates, now) -> Node``
    (see ``policies/eviction``).
    """

    def __init__(self, capacity_tokens: int, eviction, block_size: int = 16):
        self.capacity = int(capacity_tokens)
        self.block_size = block_size
        self.eviction = eviction
        self.root = Node((), None, None, 0.0)
        self.size = 0
        self.finished_records: List[Record] = []   # records of evicted nodes
        self.evictions = 0

    # ------------------------------------------------------------------ lookup
    def _split(self, node: Node, k: int) -> Node:
        """Split ``node`` after its first ``k`` tokens; return the new upper node."""
        assert 0 < k < len(node)
        parent = node.parent
        upper_record = node.record
        lower_record = upper_record.split(k, len(node)) if upper_record else None
        upper = Node(node.tokens[:k], parent, upper_record, node.last_access)
        upper.hits = node.hits
        parent.children[upper.tokens[0]] = upper
        node.tokens = node.tokens[k:]
        node.parent = upper
        node.record = lower_record
        upper.children[node.tokens[0]] = node
        return upper

    def match(self, tokens: Sequence[int], split: bool = True) -> Tuple[int, List[Node]]:
        """Longest cached prefix of ``tokens``. Splits a partially matched edge
        so the matched part is its own node (as SGLang does)."""
        node, i, path = self.root, 0, []
        n = len(tokens)
        while i < n:
            child = node.children.get(tokens[i])
            if child is None:
                break
            edge = child.tokens
            k = 0
            lim = min(len(edge), n - i)
            while k < lim and edge[k] == tokens[i + k]:
                k += 1
            if k == 0:
                break
            if k < len(edge):
                if not split:
                    i += k
                    path.append(child)
                    break
                child = self._split(child, k)
            path.append(child)
            node = child
            i += k
        return i, path

    # --------------------------------------------------------------- requests
    def request(self, agent: str, tokens: Sequence[int], now: float, insert: bool = True) -> RequestResult:
        """Serve one prompt: reuse the cached prefix, compute and cache the rest."""
        tokens = tuple(tokens)
        matched, path = self.match(tokens)
        for nd in path:
            nd.last_access = now
            nd.hits += 1
            nd.record.users[agent] = nd.record.users.get(agent, 0) + 1
        computed = len(tokens) - matched
        cached = True
        if computed and insert:
            cached = self._insert_suffix(agent, tokens, matched, path, now)
            if not cached:  # computed but not kept: still a cost to bill
                self.finished_records.append(Record(cost=computed, writer=agent, users={agent: 1}))
        return RequestResult(agent, len(tokens), matched, computed, cached)

    def _insert_suffix(self, agent: str, tokens, matched: int, path: List[Node], now: float) -> bool:
        need = len(tokens) - matched
        if need > self.capacity:
            return False
        for nd in path:
            nd.lock += 1
        try:
            while self.size + need > self.capacity:
                if not self._evict_one(now):
                    return False
        finally:
            for nd in path:
                nd.lock -= 1
        parent = path[-1] if path else self.root
        suffix = tokens[matched:]
        rec = Record(cost=len(suffix), writer=agent, users={agent: 1})
        leaf = Node(suffix, parent, rec, now)
        leaf.hits = 1
        parent.children[suffix[0]] = leaf
        self.size += len(suffix)
        return True

    # --------------------------------------------------------------- eviction
    def evictable_leaves(self) -> List[Node]:
        out, stack = [], [self.root]
        while stack:
            nd = stack.pop()
            if nd.children:
                stack.extend(nd.children.values())
            elif nd is not self.root and nd.lock == 0:
                out.append(nd)
        return out

    def _evict_one(self, now: float) -> bool:
        candidates = self.evictable_leaves()
        if not candidates:
            return False
        victim = self.eviction.choose_victim(self, candidates, now)
        self.remove(victim)
        return True

    def remove(self, node: Node) -> None:
        assert node.is_leaf and node is not self.root
        del node.parent.children[node.tokens[0]]
        self.size -= len(node)
        self.evictions += 1
        if node.record:
            self.finished_records.append(node.record)

    # ----------------------------------------------------------------- views
    def nodes(self) -> List[Node]:
        out, stack = [], list(self.root.children.values())
        while stack:
            nd = stack.pop()
            out.append(nd)
            stack.extend(nd.children.values())
        return out

    def all_records(self) -> List[Record]:
        """Records of evicted nodes plus those still cached."""
        return self.finished_records + [nd.record for nd in self.nodes() if nd.record]

    @property
    def blocks_used(self) -> int:
        return -(-self.size // self.block_size)

    def check_invariants(self) -> None:
        total = sum(len(nd) for nd in self.nodes())
        assert total == self.size, (total, self.size)
        assert self.size <= self.capacity
        for nd in self.nodes():
            assert len(nd) > 0
            assert nd.parent.children[nd.tokens[0]] is nd
