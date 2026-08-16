#!/usr/bin/env python
"""Train a Q(S,a) controller on counterfactual structural outcomes.

This script implements the audit's central recommendation: train Q(S,a)
from counterfactual outcomes instead of classifying actions from labels.

Pipeline:
  1. Generate a large counterfactual dataset across varied topology families.
  2. Train Q(S,a) = E[ΔU(S,a)] via regression.
  3. Evaluate on held-out topology families never seen during training.

Usage:
    python scripts/train_q_controller.py [--num-samples N] [--epochs E]
"""
import argparse
import json
import sys
import os

# Ensure local src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgae_v3.benchmark.counterfactual import (
    TOPOLOGY_FAMILIES,
    HELD_OUT_FAMILIES,
    generate_counterfactual_dataset,
    train_q_network,
    evaluate_q_network,
)


def main():
    parser = argparse.ArgumentParser(description="Train Q(S,a) controller on counterfactual outcomes")
    parser.add_argument("--num-samples", type=int, default=12000, help="Number of counterfactual samples to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("Q(S,a) Counterfactual Structural Controller Training")
    print("=" * 70)

    # 1. Generate training dataset
    print(f"\n[1/3] Generating {args.num_samples} counterfactual samples...")
    print(f"  Training families: {TOPOLOGY_FAMILIES}")
    train_dataset = generate_counterfactual_dataset(
        num_samples=args.num_samples,
        families=TOPOLOGY_FAMILIES,
        seed=args.seed,
    )
    print(f"  Generated {len(train_dataset)} (state, action, ΔU) triples")

    # 2. Train Q-network
    print(f"\n[2/3] Training Q-network for {args.epochs} epochs...")
    result = train_q_network(train_dataset, epochs=args.epochs, seed=args.seed)
    print(f"  Final loss: {result.losses[-1]:.6f}")
    print(f"  Training accuracy: {result.train_accuracy:.4f}")
    print(f"  Training samples: {result.train_samples}")

    # 3. Evaluate on held-out topology families
    print(f"\n[3/3] Evaluating on held-out topology families...")
    print(f"  Held-out families: {HELD_OUT_FAMILIES}")
    eval_results = evaluate_q_network(
        result.q_network,
        families=HELD_OUT_FAMILIES,
        num_states_per_family=50,
        seed=args.seed + 999,
    )

    print(f"\n{'Family':<25} {'States':>7} {'Accuracy':>10} {'Mean ΔU':>10} {'Mean Regret':>12}")
    print("-" * 70)
    for r in eval_results:
        print(f"{r.family:<25} {r.num_states:>7} {r.accuracy:>10.4f} {r.mean_delta_utility:>10.4f} {r.mean_regret:>12.4f}")

    # Also evaluate on in-distribution families for comparison
    print(f"\n  In-distribution evaluation:")
    in_dist_results = evaluate_q_network(
        result.q_network,
        families=TOPOLOGY_FAMILIES[:4],  # sample 4 training families
        num_states_per_family=50,
        seed=args.seed + 123,
    )
    print(f"\n{'Family':<25} {'States':>7} {'Accuracy':>10} {'Mean ΔU':>10} {'Mean Regret':>12}")
    print("-" * 70)
    for r in in_dist_results:
        print(f"{r.family:<25} {r.num_states:>7} {r.accuracy:>10.4f} {r.mean_delta_utility:>10.4f} {r.mean_regret:>12.4f}")

    # Summary
    in_dist_acc = sum(r.accuracy for r in in_dist_results) / max(len(in_dist_results), 1)
    held_out_acc = sum(r.accuracy for r in eval_results) / max(len(eval_results), 1)
    print(f"\n{'=' * 70}")
    print(f"Summary:")
    print(f"  In-distribution accuracy:  {in_dist_acc:.4f}")
    print(f"  Held-out accuracy:         {held_out_acc:.4f}")
    print(f"  Generalization gap:        {in_dist_acc - held_out_acc:.4f}")
    print(f"{'=' * 70}")

    if args.out:
        output = {
            "num_train_samples": len(train_dataset),
            "epochs": args.epochs,
            "final_loss": result.losses[-1],
            "train_accuracy": result.train_accuracy,
            "in_distribution": {
                r.family: {"accuracy": r.accuracy, "mean_delta_u": r.mean_delta_utility, "mean_regret": r.mean_regret}
                for r in in_dist_results
            },
            "held_out": {
                r.family: {"accuracy": r.accuracy, "mean_delta_u": r.mean_delta_utility, "mean_regret": r.mean_regret}
                for r in eval_results
            },
            "in_distribution_accuracy": in_dist_acc,
            "held_out_accuracy": held_out_acc,
            "generalization_gap": in_dist_acc - held_out_acc,
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
