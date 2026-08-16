"""Production dynamics hardening utilities for LGAE v5.3.

This module contains control state that is intentionally *not* part of the
compiled numerical kernel:

- curvature EMA / variance tracking and hysteretic surgery eligibility;
- latent-equilibrium barriers for slow topology updates;
- graph-conditioned control-variate baselines for structural credit.

All objects have deterministic state dictionaries so policy-affecting state can
be checkpointed by the owning controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import math

import torch
from torch import Tensor

from .mutations import canonical_edge


@dataclass(slots=True)
class CurvatureEMAEntry:
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    @property
    def sigma(self) -> float:
        return math.sqrt(max(0.0, float(self.variance)))


class CurvatureHysteresisController:
    """EMA-smoothed edge-curvature controller with uncertainty deadband.

    The tracker intentionally stores *edge-local* statistics.  It does not claim
    that a fast proxy such as AF3/WAF3 is an exact curvature certificate; it only
    stabilizes automatic surgery proposals so instantaneous proxy noise cannot
    cause add/prune oscillation.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        variance_alpha: float | None = None,
        min_samples: int = 3,
        sigma_guard: float = 1.0,
    ) -> None:
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must lie in (0,1]")
        va = float(alpha if variance_alpha is None else variance_alpha)
        if not (0.0 < va <= 1.0):
            raise ValueError("variance_alpha must lie in (0,1]")
        if int(min_samples) < 1:
            raise ValueError("min_samples must be positive")
        if float(sigma_guard) < 0:
            raise ValueError("sigma_guard cannot be negative")
        self.alpha = float(alpha)
        self.variance_alpha = va
        self.min_samples = int(min_samples)
        self.sigma_guard = float(sigma_guard)
        self.entries: dict[tuple[int, int], CurvatureEMAEntry] = {}

    def update(self, curvatures: Mapping[tuple[int, int], float]) -> None:
        for edge, raw in curvatures.items():
            value = float(raw)
            if not math.isfinite(value):
                continue
            key = canonical_edge(*edge)
            entry = self.entries.get(key)
            if entry is None:
                self.entries[key] = CurvatureEMAEntry(value, 0.0, 1)
                continue
            old_mean = float(entry.mean)
            new_mean = (1.0 - self.alpha) * old_mean + self.alpha * value
            # EMA of squared innovations.  Using the old mean avoids artificially
            # driving variance to zero during a fast-moving transient.
            innovation2 = (value - old_mean) ** 2
            entry.variance = (1.0 - self.variance_alpha) * float(entry.variance) + self.variance_alpha * innovation2
            entry.mean = new_mean
            entry.count += 1

    def edge_stats(self, u: int, v: int) -> CurvatureEMAEntry | None:
        return self.entries.get(canonical_edge(u, v))

    def _node_incident_stats(self, node: int) -> list[CurvatureEMAEntry]:
        n = int(node)
        return [e for (u, v), e in self.entries.items() if u == n or v == n]

    def proposal_stats(self, action: str, u: int, v: int) -> tuple[float | None, float | None, int]:
        action = str(action).lower()
        if action == "prune":
            entry = self.edge_stats(u, v)
            if entry is None:
                return None, None, 0
            return float(entry.mean), float(entry.sigma), int(entry.count)
        if action == "add":
            # A missing edge has no own curvature.  Use the strongest negative
            # incident pressure at its endpoints as a conservative bottleneck proxy.
            candidates = self._node_incident_stats(u) + self._node_incident_stats(v)
            if not candidates:
                return None, None, 0
            mature = [e for e in candidates if e.count >= self.min_samples]
            pool = mature or candidates
            chosen = min(pool, key=lambda e: e.mean)
            return float(chosen.mean), float(chosen.sigma), int(chosen.count)
        raise ValueError("action must be 'add' or 'prune'")

    def allows(
        self,
        action: str,
        u: int,
        v: int,
        *,
        add_threshold: float,
        prune_threshold: float,
    ) -> tuple[bool, dict[str, Any]]:
        if not float(add_threshold) < float(prune_threshold):
            raise ValueError("add_threshold must be below prune_threshold")
        mean, sigma, count = self.proposal_stats(action, u, v)
        details = {
            "action": str(action),
            "edge": canonical_edge(u, v),
            "ema_curvature": mean,
            "sigma_curvature": sigma,
            "samples": int(count),
            "add_threshold": float(add_threshold),
            "prune_threshold": float(prune_threshold),
        }
        if mean is None or sigma is None or count < self.min_samples:
            details["reason"] = "curvature_ema_warmup"
            return False, details

        # Require the hysteresis band itself to be wider than the local noise
        # envelope.  Otherwise neither surgery direction is trustworthy.
        band = float(prune_threshold) - float(add_threshold)
        if band <= 2.0 * self.sigma_guard * float(sigma):
            details["reason"] = "curvature_noise_exceeds_hysteresis_band"
            return False, details

        if str(action).lower() == "add":
            allowed = float(mean) < float(add_threshold)
            details["reason"] = "below_add_threshold" if allowed else "not_below_add_threshold"
            return allowed, details
        allowed = float(mean) > float(prune_threshold)
        details["reason"] = "above_prune_threshold" if allowed else "not_above_prune_threshold"
        return allowed, details

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "variance_alpha": self.variance_alpha,
            "min_samples": self.min_samples,
            "sigma_guard": self.sigma_guard,
            "entries": [
                [u, v, e.mean, e.variance, e.count]
                for (u, v), e in sorted(self.entries.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "CurvatureHysteresisController":
        obj = cls(
            alpha=float(payload.get("alpha", 0.1)),
            variance_alpha=float(payload.get("variance_alpha", payload.get("alpha", 0.1))),
            min_samples=int(payload.get("min_samples", 3)),
            sigma_guard=float(payload.get("sigma_guard", 1.0)),
        )
        for u, v, mean, var, count in payload.get("entries", []):
            obj.entries[canonical_edge(int(u), int(v))] = CurvatureEMAEntry(float(mean), float(var), int(count))
        return obj


class LatentEquilibriumBarrier:
    """Require consecutive low-drift latent steps before slow structural surgery."""

    def __init__(self, delta_tol: float = 1e-3, required_consecutive: int = 3) -> None:
        if float(delta_tol) <= 0:
            raise ValueError("delta_tol must be positive")
        if int(required_consecutive) < 1:
            raise ValueError("required_consecutive must be positive")
        self.delta_tol = float(delta_tol)
        self.required_consecutive = int(required_consecutive)
        self._previous: Tensor | None = None
        self.consecutive = 0
        self.last_relative_delta = float("inf")

    @torch.no_grad()
    def observe(self, z: Tensor) -> bool:
        current = z.detach()
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current.clone()
            self.consecutive = 0
            self.last_relative_delta = float("inf")
            return False
        prev = self._previous.to(device=current.device, dtype=current.dtype)
        denom = float(torch.linalg.vector_norm(prev).item())
        delta = float(torch.linalg.vector_norm(current - prev).item()) / max(denom, 1e-12)
        self.last_relative_delta = delta
        self.consecutive = self.consecutive + 1 if delta < self.delta_tol else 0
        self._previous = current.clone()
        return self.is_equilibrated

    @property
    def is_equilibrated(self) -> bool:
        return self.consecutive >= self.required_consecutive

    def summary(self) -> dict[str, Any]:
        return {
            "delta_tol": self.delta_tol,
            "required_consecutive": self.required_consecutive,
            "consecutive": self.consecutive,
            "last_relative_delta": self.last_relative_delta,
            "equilibrated": self.is_equilibrated,
        }


class GraphHashBaseline:
    """Low-variance structural return baseline keyed by graph-state hash buckets.

    Exact graph hashes rarely repeat in a plastic system.  Hash bucketing provides a
    deterministic tabular approximation to ``V_phi(H(G_t))`` while preserving the
    graph configuration as the conditioning source.  A counterfactual NO_OP value can
    override this estimate when available.
    """

    def __init__(self, buckets: int = 1024, ema_alpha: float = 0.10) -> None:
        if int(buckets) < 1:
            raise ValueError("buckets must be positive")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError("ema_alpha must lie in (0,1]")
        self.buckets = int(buckets)
        self.ema_alpha = float(ema_alpha)
        self.values = [0.0] * self.buckets
        self.counts = [0] * self.buckets

    def _bucket(self, graph_hash: str) -> int:
        digest = hashlib.sha256(str(graph_hash).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.buckets

    def predict(self, graph_hash: str) -> float:
        i = self._bucket(graph_hash)
        return float(self.values[i]) if self.counts[i] else 0.0

    def update(self, graph_hash: str, realized_return: float) -> None:
        value = float(realized_return)
        if not math.isfinite(value):
            return
        i = self._bucket(graph_hash)
        if self.counts[i] == 0:
            self.values[i] = value
        else:
            self.values[i] = (1.0 - self.ema_alpha) * self.values[i] + self.ema_alpha * value
        self.counts[i] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "buckets": self.buckets,
            "ema_alpha": self.ema_alpha,
            "values": list(self.values),
            "counts": list(self.counts),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "GraphHashBaseline":
        obj = cls(int(payload.get("buckets", 1024)), float(payload.get("ema_alpha", 0.1)))
        vals = list(payload.get("values", []))
        counts = list(payload.get("counts", []))
        if len(vals) == obj.buckets and len(counts) == obj.buckets:
            obj.values = [float(v) for v in vals]
            obj.counts = [int(c) for c in counts]
        return obj
