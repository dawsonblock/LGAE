from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from .types import GraphBuffers, edge_role_from_code


def graphbuffers_to_networkx(graph: GraphBuffers) -> nx.Graph:
    graph.validate()
    g = nx.Graph()
    g.add_nodes_from(range(graph.num_nodes))
    src, dst, w = graph.active()
    lengths = graph.active_length()
    roles = graph.active_roles()
    for u, v, ww, ell, rr in zip(src.tolist(), dst.tolist(), w.tolist(), lengths.tolist(), roles.tolist()):
        g.add_edge(
            int(u), int(v),
            weight=float(ww),  # affinity/conductance
            length=float(ell),  # metric length
            role=edge_role_from_code(int(rr)).value,
        )
    return g


def topology_signature(g: nx.Graph) -> dict[str, float]:
    c = nx.number_connected_components(g)
    n = g.number_of_nodes()
    e = g.number_of_edges()
    beta1 = e - n + c
    return {"nodes": float(n), "edges": float(e), "beta0": float(c), "beta1": float(beta1)}


def topology_drift(a: dict[str, float], b: dict[str, float]) -> float:
    return float(abs(a.get("beta0", 0) - b.get("beta0", 0)) + abs(a.get("beta1", 0) - b.get("beta1", 0)))


def persistent_homology_signature(z: torch.Tensor, maxdim: int = 1) -> dict[str, float] | None:
    """Persistent-homology summary of the latent cloud; None if ripser is unavailable."""
    try:
        from ripser import ripser
    except Exception:
        return None
    arr = z.detach().cpu().float().numpy()
    dgms = ripser(arr, maxdim=maxdim)["dgms"]
    out: dict[str, float] = {}
    for dim, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else dgm
        persistence = (finite[:, 1] - finite[:, 0]) if len(finite) else np.array([], dtype=float)
        out[f"ph{dim}_count"] = float(len(dgm))
        out[f"ph{dim}_total_persistence"] = float(persistence.sum()) if len(persistence) else 0.0
    return out


def persistent_homology_drift(a: dict[str, float] | None, b: dict[str, float] | None) -> float | None:
    if a is None or b is None:
        return None
    keys = sorted(set(a) | set(b))
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def persistent_homology_diagrams(z: torch.Tensor, maxdim: int = 1) -> list[np.ndarray] | None:
    """Return raw persistence diagrams from ripser; None if unavailable.

    Each diagram is an array of shape (n_points, 2) with (birth, death) pairs.
    """
    try:
        from ripser import ripser
    except Exception:
        return None
    arr = z.detach().cpu().float().numpy()
    dgms = ripser(arr, maxdim=maxdim)["dgms"]
    return [np.asarray(d) for d in dgms]


def bottleneck_distance(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Exact bottleneck distance for finite persistence diagrams.

    The bottleneck objective minimizes the *maximum* matching cost, not the
    total assignment cost. We build the standard augmented bipartite matching
    problem with diagonal copies and binary-search the finite candidate costs.
    """
    def _finite(d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if d.size == 0:
            return np.empty((0, 2), dtype=float)
        d = d.reshape(-1, 2)
        return d[np.isfinite(d).all(axis=1)]

    a = _finite(dgm_a)
    b = _finite(dgm_b)
    na, nb = len(a), len(b)
    if na == 0 and nb == 0:
        return 0.0

    def _diag_dist(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.empty((0,), dtype=float)
        return 0.5 * np.abs(pts[:, 1] - pts[:, 0])

    if na == 0:
        return float(_diag_dist(b).max(initial=0.0))
    if nb == 0:
        return float(_diag_dist(a).max(initial=0.0))

    # L-infinity pair distances.
    pair = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=-1)
    da = _diag_dist(a)
    db = _diag_dist(b)

    size = na + nb
    inf = float("inf")
    cost = np.full((size, size), inf, dtype=float)
    cost[:na, :nb] = pair
    # A points may match only their own diagonal copy.
    for i in range(na):
        cost[i, nb + i] = da[i]
    # B points may match only their own diagonal copy.
    for j in range(nb):
        cost[na + j, j] = db[j]
    # Diagonal copies can match one another at zero cost.
    cost[na:, nb:] = 0.0

    candidates = np.unique(cost[np.isfinite(cost)])
    if candidates.size == 0:
        return 0.0

    def _has_perfect_matching(threshold: float) -> bool:
        allowed = np.isfinite(cost) & (cost <= threshold + 1e-15)
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import maximum_bipartite_matching
            match = maximum_bipartite_matching(csr_matrix(allowed.astype(np.int8)), perm_type="column")
            return bool(np.all(match >= 0))
        except Exception:
            # Deterministic augmenting-path fallback. This checks feasibility,
            # which is the correct bottleneck decision problem.
            adj = [np.flatnonzero(allowed[r]).tolist() for r in range(size)]
            match_col = [-1] * size

            def dfs(r: int, seen: list[bool]) -> bool:
                for c in adj[r]:
                    if seen[c]:
                        continue
                    seen[c] = True
                    if match_col[c] < 0 or dfs(match_col[c], seen):
                        match_col[c] = r
                        return True
                return False

            for r in range(size):
                if not dfs(r, [False] * size):
                    return False
            return True

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(float(candidates[mid])):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def persistent_homology_bottleneck_drift(
    z_a: torch.Tensor, z_b: torch.Tensor, maxdim: int = 1
) -> float | None:
    """Compute the max bottleneck distance across all PH dimensions.

    Returns None if ripser is unavailable.
    """
    dgms_a = persistent_homology_diagrams(z_a, maxdim=maxdim)
    dgms_b = persistent_homology_diagrams(z_b, maxdim=maxdim)
    if dgms_a is None or dgms_b is None:
        return None
    max_bd = 0.0
    for da, db in zip(dgms_a, dgms_b):
        bd = bottleneck_distance(da, db)
        max_bd = max(max_bd, bd)
    return float(max_bd)
