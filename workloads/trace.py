"""Trace format and token-segment helpers.

A trace is a list of ``Request`` objects in time order. Prompts are built from
named *segments* (system prompt, tool specs, a document, a chat turn). The same
segment name always maps to the same token ids, so shared segments become
shared prefixes exactly as they would with a real tokenizer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Request:
    time: float
    agent: str
    tokens: Tuple[int, ...]
    kind: str = "request"          # "request" or "keepalive"


class SegmentStore:
    """Maps segment names to unique, deterministic token-id runs."""

    def __init__(self) -> None:
        self._segs: Dict[str, Tuple[int, ...]] = {}
        self._next = 1

    def get(self, name: str, length: int) -> Tuple[int, ...]:
        seg = self._segs.get(name)
        if seg is None:
            seg = tuple(range(self._next, self._next + length))
            self._next += length
            self._segs[name] = seg
        return seg

    def prompt(self, parts: Iterable[Tuple[str, int]]) -> Tuple[int, ...]:
        out: List[int] = []
        for name, length in parts:
            out.extend(self.get(name, length))
        return tuple(out)


def save_jsonl(trace: List[Request], path: str) -> None:
    with open(path, "w") as f:
        for r in trace:
            f.write(json.dumps({"time": r.time, "agent": r.agent, "kind": r.kind, "tokens": list(r.tokens)}) + "\n")


def load_jsonl(path: str) -> List[Request]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(Request(d["time"], d["agent"], tuple(d["tokens"]), d.get("kind", "request")))
    return out
