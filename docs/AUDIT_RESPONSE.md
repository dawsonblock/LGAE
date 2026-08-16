# Audit findings and response (v5.3.1)

This document records the findings of an independent audit of the LGAE
codebase and tracks which were addressed in v5.3.1 versus accepted as
longer-term research direction.

## Audit summary

The audit's central conclusion:

> LGAE can measure graph geometry much better than it can yet reason
> about how to change it.

This is accurate. The geometry/numerical layer is the strongest part of
the codebase. The structural-learning and release-integrity layers are
the weakest.

## Findings addressed in v5.3.1

### Release integrity (addressed)

- **Manifest hash mismatch**: The manifest was regenerated as the final
  step after all other artifacts were immutable. Verified with
  `generate_manifest.py --check`.
- **Version identity mixing**: `version.py`, `pyproject.toml`, all schema
  constants, README, and CHANGELOG now agree on 5.3.1.
- **Stale test count**: `release_verification.json` now reports 573 tests
  (was 559).
- **Qualification reports regenerated from installed wheel**: All three
  reports (geometry, production, policy) were regenerated from the
  installed 5.3.1 wheel, not from `PYTHONPATH=src`.
- **Wheel rebuilt**: `dist/lgae_v3-5.3.1-py3-none-any.whl` with recorded
  SHA-256.
- **Test suite run from installed wheel**: 573/573 pass from the wheel,
  not from source path.

### Benchmark and baselines (addressed in v5.3.1)

- **Nondeterministic policy qualification**: Fixed (seed before network
  init).
- **Circular benchmark utility in Task A**: Replaced with pure spectral
  gap.
- **No baseline comparison**: Added `compare_baselines.py` with random,
  spectral-heuristic, and oracle controllers.
- **"Held-out" seeds not held out**: Added `HeldOutBottleneck` and
  `HeldOutSpuriousEdge` with structurally different graphs.
- **No real-world experiment**: Added `run_real_experiment.py` (Karate
  Club). Result is negative and reported as such.

## Findings accepted as longer-term research

These are valid findings that require multi-week research/engineering
effort beyond a single release cycle.

### Structural decision intelligence (the central limitation)

- **Held-out generalization is poor**: The learned policy achieves 30%
  accuracy on structurally held-out tasks vs 60% for a spectral heuristic.
  This is the most important finding. The executive has learned the
  synthetic training distribution, not transferable structural reasoning.
- **Candidate recall bottleneck**: ADD_EDGE candidate generation caps at
  top-24 nodes. If the correct endpoint is outside the top-24, the correct
  mutation is impossible, not unlikely. Needs hierarchical retrieval.
- **Q(S,a) instead of action classification**: The executive should learn
  Q-values from counterfactual outcomes, not classify actions from labels.
  This aligns the learned system with the engine's actual semantics.
- **Benchmark too small**: 864 training samples is tiny for a model
  choosing among qualitatively different graph surgeries. Needs tens of
  thousands of randomized interventions across varied topology families.
- **Permutation equivariance**: The executive uses engineered global
  statistics and node/edge MLPs, not a graph-equivariant architecture.
  A GNN or graph transformer would be more appropriate.

### Dynamic gauge stability

- **Jacobian/Lipschitz control**: SO(d) membership of U does not imply
  stability of the state-dependent map z -> U(z). The gauge generator's
  Jacobian should be constrained (spectral normalization or bounded
  Lie-algebra magnitude).
- **Cayley conditioning**: The Cayley map becomes ill-conditioned as
  ||A|| grows. Generator norm should be clamped, or switch to matrix
  exponential above a threshold.

### Credit assignment and uncertainty

- **GraphHashBaseline too weak**: Hash-bucket value baseline destroys
  geometric similarity. Should be replaced with a feature-based or
  learned graph-state value function.
- **Information gain is retrospective**: The IG head predicts IG, but the
  benchmark doesn't test whether it drives active experimentation. Needs
  a dedicated test family.
- **Structural credit attribution**: Delayed returns with interfering
  mutations need counterfactual estimators or explicit mutation
  experiments, not just ordinary discounted returns.

### Governor semantics

- **CERTIFIED_GLOBAL vs SAMPLED_LOCAL**: The governor samples a subset of
  geometry. "Audit pass" should not collapse into one acceptance vocabulary.
  Distinguish global certification from sampled local checks.
- **Safety limits monitor-only by default**: Several diagnostics
  (`max_integral_lly_deficit`, `max_operator_discrepancy`, etc.) default
  to `None`. A production profile should define bounded thresholds.
- **Hardening features disabled by default**: `curvature_ema_enabled` and
  `equilibrium_barrier_enabled` default to `False`. Consider inverting:
  `ResearchConfig` permissive, `ProductionConfig` strict.

### Equilibrium and hysteresis

- **Equilibrium barrier is weak**: State-delta convergence does not
  imply fixed-point stability. Should combine with dynamics residual
  r_t = ||F(z_t) - z_t||.
- **Curvature hysteresis statistics are crude**: EWMA variance is not a
  calibrated uncertainty interval. Needs effective sample size, Bayesian
  filtering, or change-point awareness.

### Scalability

- **NetworkX in runtime path**: Repeated graph-to-NetworkX conversion
  will bottleneck at large N. Should move to tensor/CSR-native
  implementations for runtime-critical operations.
- **No scale qualification**: Need empirical complexity curves at
  N = 10^2, 10^3, 10^4, 10^5 with controlled mean degree.
- **torch.compile not performance-qualified**: Need measurements for
  eager/compiled, CPU/CUDA, across graph sizes.

### Cryptographic integrity

- **Checkpoint lacks cryptographic envelope**: SHA-256 consistency checks
  prove internal consistency, not provenance. Needs Merkle root +
  optional signing.
- **Receipt chain is tamper-evident, not identity-authenticated**: Needs
  Ed25519 or equivalent signatures for authority identity.

## Recommended next development cycle

Per the audit's recommendation, the next work should focus on
**structural decision intelligence**, not additional curvature operators.
The geometry side already has enough sensors. The controller does not yet
know what to do with them outside the benchmark distribution.

Priority order:
1. Replace GraphHashBaseline with feature-based value approximation.
2. Replace top-24 candidate generation with hierarchical retrieval.
3. Build a large counterfactual structural dataset (varied topology
   families, sizes, densities).
4. Train Q(S,a) from counterfactual outcomes instead of action
   classification.
5. Evaluate on topology families never encountered during training.
6. Add dynamic-gauge Jacobian/norm control.
7. Upgrade equilibrium gating from state-delta to dynamics residual.
8. Move toward model-predictive structural control.
9. Only after that consider calling the structural executive
   production-qualified.
