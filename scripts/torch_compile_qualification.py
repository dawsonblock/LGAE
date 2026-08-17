#!/usr/bin/env python
"""torch.compile performance qualification for LGAE.

The audit found that torch.compile is not performance-qualified. This
script measures eager vs compiled diffusion performance across graph
sizes, recording compilation time, speedup, and any failures.

Usage:
    python scripts/torch_compile_qualification.py [--out results.json]
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


def generate_graph(n: int, mean_degree: int = 4, seed: int = 0) -> nx.Graph:
    rng = np.random.RandomState(seed)
    p = mean_degree / max(n - 1, 1)
    G = nx.erdos_renyi_graph(n, p, seed=rng)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def benchmark_diffusion(n: int, compiled: bool = False, warmup: int = 3, iters: int = 10) -> dict:
    """Benchmark diffusion at a given graph size."""
    G = generate_graph(n, seed=0)
    n_actual = G.number_of_nodes()
    e_actual = G.number_of_edges()

    cfg = LGAEConfig()
    cfg.fiber.d_base = 8
    cfg.fiber.d_max = 16
    cfg.fiber.gauge_dim = 4
    cfg.audit.bakry_nodes = min(8, n_actual)
    cfg.audit.exact_lly_top_k = min(8, e_actual)

    capacity = max(256, e_actual * 2)
    graph = make_bucketed_graph_buffers(n_actual, list(G.edges()), bucket_size=capacity)
    engine = LGAEEngine(graph, cfg)

    if compiled:
        try:
            engine.diffuse_ = torch.compile(engine.diffuse_, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            return {"n": n_actual, "compiled": True, "error": f"compile setup failed: {e}"}

    # Warmup
    t0 = time.perf_counter()
    for _ in range(warmup):
        try:
            engine.diffuse_(eta=0.01)
        except Exception as e:
            return {"n": n_actual, "compiled": compiled, "error": f"warmup failed: {e}"}
    warmup_time = time.perf_counter() - t0

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        engine.diffuse_(eta=0.01)
    benchmark_time = (time.perf_counter() - t0) / iters

    return {
        "n": n_actual,
        "e": e_actual,
        "compiled": compiled,
        "warmup_time_s": warmup_time,
        "per_iter_ms": benchmark_time * 1000,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="torch.compile performance qualification")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200],
                        help="Graph sizes to benchmark")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per benchmark (seconds)")
    args = parser.parse_args()

    print("=" * 70)
    print("torch.compile Performance Qualification")
    print("=" * 70)
    print(f"\nGraph sizes: {args.sizes}")
    print(f"Timeout: {args.timeout}s per benchmark")

    results = []

    for n in args.sizes:
        print(f"\n--- N={n} ---")

        # Eager baseline
        print(f"  Eager: ", end="", flush=True)
        try:
            eager = benchmark_diffusion(n, compiled=False)
            print(f"{eager['per_iter_ms']:.2f} ms/iter")
            results.append(eager)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"n": n, "compiled": False, "error": str(e)})
            continue

        # Compiled
        print(f"  Compiled: ", end="", flush=True)
        try:
            compiled = benchmark_diffusion(n, compiled=True)
            if compiled.get("error"):
                print(f"ERROR: {compiled['error']}")
            else:
                print(f"{compiled['per_iter_ms']:.2f} ms/iter (warmup: {compiled['warmup_time_s']:.1f}s)")
                speedup = eager["per_iter_ms"] / compiled["per_iter_ms"] if compiled["per_iter_ms"] > 0 else 0
                print(f"  Speedup: {speedup:.2f}x")
                compiled["speedup"] = speedup
            results.append(compiled)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"n": n, "compiled": True, "error": str(e)})

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"\n{'N':>8} {'Eager (ms)':>12} {'Compiled (ms)':>14} {'Speedup':>10} {'Status':>10}")
    print("-" * 60)
    for i in range(0, len(results), 2):
        eager = results[i] if i < len(results) else {}
        compiled = results[i + 1] if i + 1 < len(results) else {}
        n = eager.get("n", "?")
        e_ms = eager.get("per_iter_ms", None)
        c_ms = compiled.get("per_iter_ms", None)
        speedup = compiled.get("speedup", None)
        status = "OK" if c_ms is not None and not compiled.get("error") else "FAIL"
        e_str = f"{e_ms:.2f}" if e_ms else "N/A"
        c_str = f"{c_ms:.2f}" if c_ms else "N/A"
        s_str = f"{speedup:.2f}x" if speedup else "N/A"
        print(f"{n:>8} {e_str:>12} {c_str:>14} {s_str:>10} {status:>10}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
