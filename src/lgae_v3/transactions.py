"""Atomic graph/cache transaction helpers.

The authoritative engine normally mutates a shadow graph and therefore never exposes
partial state to an ANN cache. This context manager is provided for external/in-place
workflows: graph tensors are restored atomically on rollback and attached neighbor
indices are generation-invalidated so stale leaves are rebuilt lazily on next query.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from .types import GraphBuffers


class GraphTransaction:
    def __init__(self, graph: GraphBuffers, indices: Iterable[Any] = ()) -> None:
        self.graph = graph
        self.indices = list(indices)
        self._backup: GraphBuffers | None = None
        self._committed = False
        self._rolled_back = False

    def __enter__(self) -> "GraphTransaction":
        self._backup = self.graph.clone()
        return self

    @torch.no_grad()
    def rollback(self) -> None:
        if self._backup is None or self._rolled_back:
            return
        b = self._backup
        self.graph.src.copy_(b.src)
        self.graph.dst.copy_(b.dst)
        self.graph.weight.copy_(b.weight)
        self.graph.valid.copy_(b.valid)
        if self.graph.length is not None and b.length is not None:
            self.graph.length.copy_(b.length)
        if self.graph.role is not None and b.role is not None:
            self.graph.role.copy_(b.role)
        if self.graph.slot_generation is not None and b.slot_generation is not None:
            self.graph.slot_generation.copy_(b.slot_generation)
        self.graph.version = int(b.version)
        for index in self.indices:
            if hasattr(index, "invalidate"):
                index.invalidate(graph_version=int(self.graph.version), reason="transaction_rollback")
            elif hasattr(index, "mark_dirty"):
                index.mark_dirty(graph_version=int(self.graph.version), reason="transaction_rollback")
        self._rolled_back = True

    def commit(self) -> None:
        self.graph.validate()
        self._committed = True
        for index in self.indices:
            if hasattr(index, "invalidate"):
                index.invalidate(graph_version=int(self.graph.version), reason="transaction_commit")
            elif hasattr(index, "mark_dirty"):
                index.mark_dirty(graph_version=int(self.graph.version), reason="transaction_commit")

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False


def graph_transaction(graph: GraphBuffers, *indices: Any) -> GraphTransaction:
    return GraphTransaction(graph, indices)
