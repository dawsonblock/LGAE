#!/usr/bin/env python
"""Scale qualification: empirical complexity curves for LGAE operations.

The audit found no empirical complexity curves. This script measures
diffusion, audit, mutation, and checkpoint times across graph sizes
from N=10^2 to N=10^4 with controlled mean degree.

Usage:
    python scripts/scale_qualification.py [--out results.json]
"""
import argparse
import json
import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import networkx as nx

from lgae_v3 import LGAEConfig, LGAEEngine, make_bucketed_graph_buffers
from lgae_v3.mutations import AddEdge


def generate_graph(n: int, mean_degree: int, seed: int) -> nx.Graph:
    """Generate a connected random graph with approximately the given mean degree."""
    rng = np.random.RandomState(seed)
    p = mean_degree / max(n - 1, 1)
    G = nx.erdos_renyi_graph(n, p, seed=rng)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def benchmark_size(n: int, mean_degree: int = 4, seed: int = 0) -> dict:
    """Benchmark LGAE operations at a given graph size."""
    G = generate_graph(n, mean_degree, seed)
    n_actual = G.number_of_nodes()
    e_actual = G.number_of_edges()

    cfg = LGAEConfig()
    cfg.fiber.d_base = 8
    cfg.fiber.d_max = 16
    cfg.fiber.gauge_dim = 4
    # Reduce audit sampling for large graphs to avoid timeouts
    cfg.audit.bakry_nodes = min(8, n_actual)
    cfg.audit.exact_lly_top_k = min(8, e_actual)
    cfg.audit.entropic_nodes = min(16, n_actual)
    cfg.audit.cde_nodes = min(4, n_actual)

    capacity = max(256, e_actual * 2)
    edges = list(G.edges())
    graph = make_bucketed_graph_buffers(n_actual, edges, bucket_size=capacity)

    engine = LGAEEngine(graph, cfg)

    # Warm up
    engine.diffuse_(eta=0.01)

    # 1. Diffusion time
    t0 = time.perf_counter()
    for _ in range(5):
        engine.diffuse_(eta=0.01)
    t_diffuse = (time.perf_counter() - t0) / 5

    # 2. Audit time
    t0 = time.perf_counter()
    engine.audit()
    t_audit = time.perf_counter() - t0

    # 3. Mutation evaluation time (shadow + governor)
    t0 = time.perf_counter()
    non_edges = list(nx.non_edges(G))
    if non_edges:
        u, v = non_edges[0]
        engine.evaluate_and_maybe_commit(AddEdge(u, v))
    t_mutation = time.perf_counter() - t0

    # 4. Authority hash time
    t0 = time.perf_counter()
    engine.authority_hash()
    t_hash = time.perf_counter() - t0

    # 5. Memory (approximate)
    import resource
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "n": n_actual,
        "e": e_actual,
        "mean_degree": float(2 * e_actual) / max(n_actual, 1),
        "t_diffuse_ms": t_diffuse * 1000,
        "t_audit_ms": t_audit * 1000,
        "t_mutation_ms": t_mutation * 1000,
        "t_hash_ms": t_hash * 1000,
        "mem_kb": mem_kb,
    }


def main():
    parser = argparse.ArgumentParser(description="Scale qualification for LGAE")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000, 3000],
                        help="Graph sizes to benchmark")
    parser.add_argument("--mean-degree", type=int, default=4, help="Target mean degree")
    args = parser.parse_args()

    print("=" * 70)
    print("LGAE Scale Qualification")
    print("=" * 70)
    print(f"\nGraph sizes: {args.sizes}")
    print(f"Mean degree: {args.mean_degree}")

    results = []
    print(f"\n{'N':>8} {'E':>8} {'t_diff':>10} {'t_audit':>10} {'t_mut':>10} {'t_hash':>10} {'mem_KB':>10}")
    print("-" * 70)

    for n in args.sizes:
        try:
            r = benchmark_size(n, args.mean_degree)
            results.append(r)
            print(f"{r['n']:>8} {r['e']:>8} {r['t_diffuse_ms']:>10.2f} {r['t_audit_ms']:>10.2f} "
                  f"{r['t_mutation_ms']:>10.2f} {r['t_hash_ms']:>10.2f} {r['mem_kb']:>10}")
        except Exception as e:
            print(f"{n:>8} ERROR: {e}")
            results.append({"n": n, "error": str(e)})

    # Compute scaling exponents
    print(f"\n{'=' * 70}")
    print("Scaling analysis:")
    valid = [r for r in results if "t_diffuse_ms" in r and r["n"] > 100]
    if len(valid) >= 2:
        ns = [math.log(r["n"]) for r in valid]
        for op in ["t_diffuse_ms", "t_audit_ms", "t_mutation_ms", "t_hash_ms"]:
            ts = [math.log(max(r[op], 0.001)) for r in valid]
            # Linear regression: log(t) = a * log(n) + b
            n_mean = sum(ns) / len(ns)
            t_mean = sum(ts) / len(ts)
            num = sum((n - n_mean) * (t - t_mean) for n, t in zip(ns, ts))
            den = sum((n - n_mean) ** 2 for n in ns)
            exponent = num / den if den else 0
            print(f"  {op}: O(N^{exponent:.2f})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
