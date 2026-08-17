# LGAE — Complete Guide

**Version:** 5.3.3
**Repository:** [dawsonblock/LGAE](https://github.com/dawsonblock/LGAE)
**Package:** `lgae-v3` (pip) / `lgae_v3` (import)
**Tests:** 652 passing
**Source files:** 57
**Test files:** 33
**License:** MIT

---

## Table of Contents

1. [What LGAE Is](#1-what-lgae-is)
2. [Architecture](#2-architecture)
3. [The Four Layers](#3-the-four-layers)
4. [Numerical Geometry](#4-numerical-geometry)
5. [Structural Executive](#5-structural-executive)
6. [Governor and Qualification](#6-governor-and-qualification)
7. [Safety Architecture](#7-safety-architecture)
8. [Cryptographic Integrity](#8-cryptographic-integrity)
9. [Benchmark and Evaluation](#9-benchmark-and-evaluation)
10. [Q-Learning Controller](#10-q-learning-controller)
11. [Advanced Control](#11-advanced-control)
12. [Audit: Findings and Responses](#12-audit-findings-and-responses)
13. [Honest Limitations](#13-honest-limitations)
14. [File Map](#14-file-map)
15. [Verification and Release](#15-verification-and-release)
16. [Version History](#16-version-history)

---

## 1. What LGAE Is

LGAE (Laplacian Geometric Adaptive Evolution) is a research system that
combines **differential geometry on graphs** with **learned structural
control**. It measures graph geometry (curvature, spectral gaps,
persistent homology) and attempts to decide whether and how to modify
the graph's structure to improve a utility function.

The core principle:

> **Field dynamics are sparse and compiled; discrete evolution is
> transactional and eager; curvature diagnoses rather than directly
> dictates topology.**

LGAE does **not** claim to be a production-qualified autonomous control
system. It is a research codebase with strong numerical foundations,
honest benchmarking, and clearly documented limitations.

### Package identity

The release is **LGAE v5.3.3**. The Python distribution/module is
`lgae-v3` / `lgae_v3` — a historical name kept for import stability.
`pip install lgae-v3` gives you `import lgae_v3` at version `5.3.3`.
The CLI is `lgae-v3`.

---

## 2. Architecture

The system is organized as four interacting layers, each with distinct
responsibilities and timescales:

```
┌─────────────────────────────────────────────────────────┐
│                    LGAEEngine                            │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐ │
│  │ Layer 1  │   │ Layer 2  │   │ Layer 3  │   │ L4   │ │
│  │ Contin.  │──▶│ Geometric│──▶│ Discrete │──▶│ Gov. │ │
│  │ Fiber    │   │ Diagnos. │   │ Mutations│   │ Qual │ │
│  │ Dynamics │   │          │   │          │   │      │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┘ │
│       │              │              │              │     │
│    diffuse_()     audit()      propose()     evaluate() │
│       │              │              │              │     │
│       ▼              ▼              ▼              ▼     │
│    z_{t+1}        curvature    candidate       accept/  │
│    (fast)         (medium)     mutations       reject   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The control loop:

```
z_t, G_t
  → geometric observation (curvature, spectral gap, homology)
  → candidate intervention (add/prune/reweight/spawn)
  → counterfactual/shadow evaluation
  → uncertainty gate
  → governor (audit + accept/reject/quarantine)
  → G_{t+1}
```

**Key principle:** Structural changes do not directly become
authoritative from the learned network. They pass through
`LGAEEngine`, shadow evaluation, and `GeometryGovernor`.

---

## 3. The Four Layers

### Layer 1: Continuous latent/fiber dynamics

- **Module:** `evolution.py`, `fibers.py`, `sheaf_diffusion.py`,
  `dynamic_gauge.py`, `operators.py`
- **Timescale:** Fast (every step)
- **What it does:** Diffuses latent states `z` over the graph using
  sheaf Laplacians with SO(d) gauge connections. Each edge carries a
  rotation matrix `U_ij` that transports fiber values between nodes.
- **Key invariant:** `U_ij ∈ SO(d)`, enforced by skew-symmetric
  generator `A_ij ∈ so(d)` and Cayley/matrix-exponential maps.

### Layer 2: Geometric diagnostics

- **Module:** `governor.py`, `operators.py`, `topology.py`,
  `uncertainty.py`
- **Timescale:** Medium (every few steps)
- **What it does:** Measures graph geometry:
  - Ollivier Ricci curvature (exact LP or Sinkhorn)
  - Bakry–Émery / Γ-calculus
  - LLY curvature
  - Spectral gap (exact or LOBPCG)
  - Persistent homology (Betti numbers, bottleneck distance)
  - Bridge detection
  - Topology signatures (β₀, β₁)

### Layer 3: Discrete structural mutations

- **Module:** `executive.py`, `mutations.py`, `structural_loop.py`
- **Timescale:** Slow (when equilibrium is reached)
- **What it does:** Proposes structural changes:
  - `ADD_EDGE` — add a new edge
  - `PRUNE_EDGE` — remove an existing edge
  - `REWEIGHT_AFFINITY` / `REWEIGHT_LENGTH` — change edge weights
  - `SPAWN_FIBER` / `PRUNE_FIBER` — change fiber dimension
  - `CHANGE_GAUGE` — change gauge connection
  - `COUPLED_REWEIGHT` — jointly reweight affinity and length
  - `NO_OP` — do nothing

### Layer 4: Governance and qualification

- **Module:** `governor.py`, `production_dynamics.py`, `credit.py`,
  `receipts.py`, `transactions.py`
- **Timescale:** Slow (per-mutation)
- **What it does:** Audits proposed mutations, accepts/rejects/
  quarantines them, tracks outcomes, and maintains a hash-chained
  receipt ledger.

---

## 4. Numerical Geometry

The numerical geometry layer is the strongest part of the codebase.
All operators are implemented with exact or certified-bounded methods.

### SO(d) gauge transport

Each edge `(i,j)` carries a rotation matrix `U_ij ∈ SO(d)` that
transports fiber values between nodes. The generator `A_ij` is
skew-symmetric (`A_ji = -A_ij`), and the map to SO(d) uses either
the Cayley transform or matrix exponential.

**v5.3.2 additions:**
- Generator Frobenius norm clamping (`generator_norm_max`, default 1.0)
  applied after antisymmetrization, preserving `A_ji = -A_ij` exactly.
- Optional spectral normalization on the gauge MLP (off by default
  due to power-iteration non-determinism with small batches).

### Curvature operators

| Operator | Method | Module |
|----------|--------|--------|
| Ollivier Ricci | Exact LP or log-domain Sinkhorn | `operators.py` |
| Bakry–Émery | Γ-calculus | `operators.py` |
| LLY | Li–Liu–Yau formula | `operators.py` |
| CDE | Coarse diffusion entropy | `operators.py` |
| AF3/WAF3 | Fast proxy | `operators.py` |

### Spectral certification

- Exact eigendecomposition for small graphs
- Sparse LOBPCG for large graphs (threshold: 256 nodes)
- Spectral gap λ₂ used as connectivity certificate

### Persistent homology

- Betti number computation
- Bottleneck distance for topology drift
- Bridge protection (Tarjan algorithm, tensor-native)

---

## 5. Structural Executive

The structural executive is the learned decision-maker. It observes
the graph's geometric state and proposes structural mutations.

### Observation

The executive computes a `StructuralObservation` from the graph and
latent state, including:
- Spectral gap, curvature statistics
- Degree moments, latent variance
- Betti numbers, topology drift
- Information gain estimate

### Candidate generation (v5.3.2: hierarchical retrieval)

The previous implementation took top-24 nodes by learned score and
enumerated all non-edges within that set (capped at 256 pairs). This
created a severe recall bottleneck: if the correct endpoint was outside
the top-24, the correct mutation was **impossible**.

The new approach has two stages:
1. **Score-based retrieval:** top-64 nodes by learned node score
2. **Latent-distance KNN retrieval:** for each top-64 node, find its
   4 nearest non-adjacent neighbors in latent space

The two pools are merged, deduplicated, and capped at 512 pairs.

### Action heads

The `ExecutiveNetwork` produces:
- `delta_u` — predicted utility change per action
- `ig` — predicted information gain per action
- `cost` — predicted cost per action
- `risk` — predicted risk per action
- `uncertainty` — epistemic uncertainty per action
- `policy_logits` — action selection logits

### Permutation-equivariant executive (v5.3.2)

New `EquivariantExecutiveNetwork` uses message-passing GNN with mean
pooling. Graph-level output is **exactly** permutation-invariant
(verified to 1e-4). Node embeddings are permutation-equivariant.

### Credit assignment

`MutationCreditTracker` in `credit.py` tracks mutation receipts and
long-term outcomes:
- Discounted return: `R = Σ γ^τ ΔU_{t+τ}`
- Advantage: `A = R - V(G_t)` using a baseline estimator
- **v5.3.2:** `GraphFeatureBaseline` replaces `GraphHashBaseline` —
  uses online ridge regression on 16-dim structural features instead
  of hash-bucket tabular approximation

---

## 6. Governor and Qualification

### GeometryGovernor

The governor audits proposed mutations before they are committed:

1. **Local mutation gate** — fast checks (bridge protection, capacity)
2. **Shadow evaluation** — apply mutation to a copy, measure geometry
3. **Multi-horizon certification** — evaluate at multiple time horizons
4. **Decision** — ACCEPT, QUARANTINE, or REJECT

### Certification levels (v5.3.2)

Mutation results now include `certification_level` in metadata:

| Level | Meaning |
|-------|---------|
| `CERTIFIED_GLOBAL` | All edges/nodes audited (small graphs only) |
| `SAMPLED_LOCAL` | Subset audited, rest extrapolated |
| `HEURISTIC_PROXY` | Only fast proxy diagnostics used |

### Mutation authority levels (v5.3.2)

Mutations are classified by risk:

| Level | Examples | Evidence threshold |
|-------|----------|-------------------|
| `REVERSIBLE` | Reweighting, Ricci flow | Low |
| `STRUCTURAL` | Add/prune edges | Medium |
| `IRREVERSIBLE` | Fiber spawn/prune, gauge changes | High |

### Equilibrium barrier (v5.3.2: dynamics residual)

The barrier prevents structural surgery while the latent dynamics are
still moving. The upgraded check requires **both**:

- State delta: `‖z_t - z_{t-1}‖ / ‖z_{t-1}‖ < δ_tol`
- Dynamics residual: `‖F(z_t) - z_t‖ / ‖z_t‖ < residual_tol`

The dynamics residual catches periodic orbits and metastable plateaus
that the state-delta check alone misses.

### Curvature hysteresis (v5.3.2: Bayesian)

The `CurvatureHysteresisController` stabilizes automatic surgery
proposals so instantaneous proxy noise cannot cause add/prune
oscillation. Upgraded from crude EWMA variance to **Normal-Inverse-Gamma
conjugate posterior** with:
- Calibrated credible intervals
- Effective sample size tracking
- Student-t predictive distribution

### Configuration profiles (v5.3.2)

| Profile | Hardening | Safety limits | Use case |
|---------|-----------|----------------|----------|
| `ResearchConfig()` | Disabled | Monitor-only (None) | Experimentation |
| `ProductionConfig()` | Enabled | Bounded thresholds | Production |
| `LGAEConfig()` | Disabled | Monitor-only | Backward compatible |

---

## 7. Safety Architecture

### Transactional mutations

All structural mutations are transactional:
1. **Shadow graph** — mutation applied to a copy
2. **Audit** — geometry checked on the shadow
3. **Commit or rollback** — original graph modified only if accepted
4. **Quarantine** — uncertain mutations held for later re-evaluation

### Fail-closed numerical behavior

All numerical checks fail closed: if a computation produces NaN, Inf,
or a shape mismatch, the mutation is rejected rather than committed.

### Betti-number protection

The governor tracks Betti numbers (β₀ = connected components, β₁ =
independent cycles) and rejects mutations that would change them
beyond configured thresholds.

### Bridge protection

Pruning a bridge edge (whose removal disconnects the graph) is blocked
by the local mutation gate. Uses tensor-native Tarjan algorithm (no
NetworkX conversion).

### Cooldown tracking

`MutationCooldownTracker` prevents rapid repeated mutations of the
same type, avoiding oscillation.

---

## 8. Cryptographic Integrity

### Checkpoint format

Two checkpoint formats:
- **Legacy `.pt`** — pickle-based (trusted local use only)
- **Safe `.ckpt/`** — safetensors + JSON (untrusted interchange)

The safe format includes:
- `tensors.safetensors` — all tensor data
- `graph.json` — graph metadata
- `controller.json` — controller state
- `governance.json` — config and authority hashes
- `manifest.json` — file list, tensor keys, **Merkle root** (v5.3.2)

### Merkle root (v5.3.2)

The manifest includes a Merkle root over all checkpoint files:
```
leaf_i = SHA-256(file_i)
merkle_root = SHA-256(SHA-256(leaf_0 || leaf_1) || SHA-256(leaf_2 || leaf_3) || ...)
```

This provides cryptographic provenance — proof that the checkpoint was
produced by someone with this exact state — without requiring a trusted
signature.

### Receipt chain

Every committed mutation gets a receipt in a JSONL ledger:
- Hash-chained: `H_i = SHA-256(H_{i-1} || R_i)`
- Binds: graph state, gauge connections, fiber state, governance config
- Tamper-evident: any modification breaks the chain

### Ed25519 signing (v5.3.2)

Receipts can optionally be signed with Ed25519:
- `signing_key` parameter in `mutation_receipt()` and `append_receipt()`
- `public_key` parameter in `verify_receipt_chain()`
- Requires `cryptography` or `PyNaCl` package
- Unsigned receipts remain backward compatible

---

## 9. Benchmark and Evaluation

### Synthetic tasks

Seven benchmark tasks with known-optimal structural changes:

| Task | Description | Correct action |
|------|-------------|----------------|
| A | Long-range bottleneck | ADD_EDGE |
| B | Local representational complexity | SPAWN_FIBER |
| C | Noisy spurious edge | PRUNE_EDGE |
| D | Coordinate-frame mismatch | CHANGE_GAUGE |
| E | Distribution shift | SPAWN_FIBER + consolidate |
| F | Nothing wrong | NO_OP |
| G | Hidden structural uncertainty (v5.3.2) | ADD_EDGE or NO_OP |

### Held-out variants

`HeldOutBottleneck` and `HeldOutSpuriousEdge` generate structurally
different graphs (different sizes, cluster splits, spurious-edge
positions) so held-out evaluation measures something beyond seed noise.

### Baselines

| Controller | Description |
|-----------|-------------|
| `RandomActionController` | Uniform random action selection |
| `SpectralHeuristicController` | Non-learned spectral/geometric rules |
| `OracleController` | Always picks the correct action (upper bound) |

### Policy qualification

`qualify_structural_policy()` trains the executive on the synthetic
tasks and evaluates on held-out seeds. Current result:
- **Diagnosis accuracy: 100%** (deterministic)
- **Mean regret: 0.0**
- Thresholds: accuracy ≥ 80%, regret ≤ 0.35

### Real-world experiment

`scripts/run_real_experiment.py` runs the governor on the Karate Club
graph. Results (honestly negative):

| Controller | Accuracy |
|-----------|----------|
| Raw (no mutation) | 70.59% |
| Random add | 73.53% |
| Spectral heuristic | 79.41% |
| LGAE governed | 67.65% |

The governor committed zero mutations on the small Karate Club graph.
This is reported as a negative result, not hidden.

---

## 10. Q-Learning Controller

### The central limitation

The audit's central conclusion:

> LGAE can measure graph geometry much better than it can yet reason
> about how to change it.

The original executive classified actions from labels. The Q-learning
controller learns `Q(S,a) = E[ΔU(S,a)]` from counterfactual outcomes
instead.

### Counterfactual dataset

`src/lgae_v3/benchmark/counterfactual.py` generates
`(state, action, ΔU)` triples by:
1. Sampling a random graph from a topology family
2. Enumerating bounded candidate interventions
3. Executing each in a shadow graph
4. Measuring Δutility (spectral gap change)

**Training families:** path, cycle, grid, star, barabasi-albert,
watts-strogatz, random, complete-bipartite

**Held-out families:** wheel, lollipop, caveman

### Q-network

Simple MLP: `observation (16-dim) → hidden (128) → Q-values (9 actions)`

Trained via MSE regression: `L = (Q(S,a) - ΔU)²`

Policy derived as `π(S) = argmax_a Q(S,a)`

### Results

| Evaluation | Accuracy |
|-----------|----------|
| In-distribution | 43% |
| Held-out | 86% |

The Q-network does better on held-out families than in-distribution
because some in-distribution families (star, cycle) have zero ΔU for
all actions, making the correct action ambiguous. Wheel and caveman
have clearer structural interventions.

### Runner script

```bash
python scripts/train_q_controller.py --num-samples 12000 --epochs 50
```

---

## 11. Advanced Control

### Model-predictive structural control (MPC)

`src/lgae_v3/mpc.py` implements receding-horizon control:

1. Enumerate candidate mutation sequences up to horizon H
2. Simulate each sequence using shadow evaluation
3. Select the sequence with highest cumulative utility
4. Execute only the first mutation (receding horizon)

Bounded by `max_branching` (default 8) and `max_sequences` (default 64).

### Scale qualification

`scripts/scale_qualification.py` measures empirical complexity curves:

| Operation | Scaling |
|-----------|---------|
| Diffusion | O(N^0.63) — sublinear (sparse) |
| Audit | Noisy (exact LP Ollivier on small samples) |
| Mutation | Noisy (depends on audit) |
| Hash | O(N^0.16) — near-constant |

### torch.compile qualification

`scripts/torch_compile_qualification.py` measures eager vs compiled:

**Result: compiled is 0.05x speedup (slower)**

Causes:
- Graph breaks from `.item()` calls in validation
- Recompilation from `step_index` integer changes
- Compilation overhead (13.2s warmup) exceeds any per-step savings

This is an honest negative result. The compiled-kernel architecture is
inherited from v5.2 but is not newly performance-qualified.

---

## 12. Audit: Findings and Responses

An independent audit identified findings across nine categories. All
are now addressed.

### Release integrity (fixed in v5.3.1)

- ✅ Manifest hash mismatch — regenerated as final step
- ✅ Version identity mixing — all agree on 5.3.2
- ✅ Stale test count — reports 652
- ✅ Qualification from installed wheel — not PYTHONPATH
- ✅ Wheel rebuilt with recorded SHA-256

### Benchmark and baselines (fixed in v5.3.1)

- ✅ Nondeterministic policy qualification — seed before init
- ✅ Circular benchmark utility — replaced with spectral gap
- ✅ No baseline comparison — added random, spectral, oracle
- ✅ "Held-out" seeds not held out — added structurally different variants
- ✅ No real-world experiment — Karate Club (negative result reported)

### Structural decision intelligence (implemented in v5.3.2)

- ✅ Held-out generalization poor — Q-learning achieves 86% on held-out
- ✅ Candidate recall bottleneck — hierarchical retrieval (top-64 + KNN)
- ✅ Q(S,a) instead of classification — counterfactual Q-learning
- ✅ Benchmark too small — 14,400 counterfactual samples across 8 families
- ✅ Permutation equivariance — GNN-based equivariant executive

### Dynamic gauge stability (implemented in v5.3.2)

- ✅ Jacobian/Lipschitz control — generator norm clamping + optional spectral norm
- ✅ Cayley conditioning — norm clamp prevents ill-conditioning

### Credit assignment (implemented in v5.3.2)

- ✅ GraphHashBaseline too weak — replaced with GraphFeatureBaseline
- ✅ Information gain not tested — Task G added
- ✅ Structural credit attribution — counterfactual estimators in Q-learning

### Governor semantics (implemented in v5.3.2)

- ✅ CERTIFIED_GLOBAL vs SAMPLED_LOCAL — certification levels in metadata
- ✅ Safety limits monitor-only — ProductionConfig sets bounded thresholds
- ✅ Hardening disabled by default — ProductionConfig enables all

### Equilibrium and hysteresis (implemented in v5.3.2)

- ✅ Equilibrium barrier weak — upgraded to dynamics residual
- ✅ Curvature hysteresis crude — upgraded to Bayesian NIG posterior

### Scalability (implemented in v5.3.2)

- ✅ NetworkX in runtime path — tensor-native topology and bridge detection
- ✅ No scale qualification — empirical complexity curves measured
- ✅ torch.compile not qualified — measured (honest negative result)

### Cryptographic integrity (implemented in v5.3.2)

- ✅ Checkpoint lacks cryptographic envelope — Merkle root added
- ✅ Receipt chain not identity-authenticated — Ed25519 signing added

---

## 13. Honest Limitations

These limitations are documented, not hidden:

1. **No real-world generalization claim.** The structural-policy
   qualification uses synthetic tasks. The Karate Club experiment is
   negative (governor commits zero mutations).

2. **Q-network's 86% held-out accuracy is on synthetic spectral-gap
   tasks.** This is not evidence of real-world structural reasoning.

3. **torch.compile is slower than eager.** The compiled-kernel
   architecture exists but is not performance-qualified. Graph breaks
   and recompilation eliminate any benefit.

4. **The spectral heuristic beats the learned policy on the Karate
   Club graph** (79.41% vs 67.65%). The learned executive does not yet
   beat simple non-learned heuristics on real-world data.

5. **The governor samples geometry, not certifies it globally.**
   `SAMPLED_LOCAL` is the typical certification level for large graphs.
   "Audit pass" does not imply "global graph safe."

6. **No deployment-safety proof.** The governor's transactional
   shadow/rollback/quarantine and fail-closed numerical behavior are
   engineering safeguards, not a formal safety argument.

7. **The structural executive is not production-qualified.** It is a
   research system with honest benchmarks and clearly documented
   limitations.

---

## 14. File Map

### Core source (`src/lgae_v3/`)

| File | Purpose |
|------|---------|
| `evolution.py` | LGAEEngine — main control loop, checkpoints |
| `executive.py` | StructuralExecutive — learned decision-maker |
| `governor.py` | GeometryGovernor — audit and accept/reject |
| `config.py` | LGAEConfig, ProductionConfig, ResearchConfig |
| `types.py` | GraphBuffers, MutationResult, CertificationLevel |
| `mutations.py` | AddEdge, PruneEdge, ReweightEdge, etc. |
| `operators.py` | Curvature operators (Ollivier, Bakry-Émery, LLY) |
| `fibers.py` | Fiber management (spawn/prune) |
| `dynamic_gauge.py` | SO(d) gauge transport with norm clamping |
| `sheaf_diffusion.py` | Sheaf Laplacian diffusion |
| `production_dynamics.py` | Equilibrium barrier, curvature hysteresis, baselines |
| `credit.py` | MutationCreditTracker, GraphFeatureBaseline |
| `receipts.py` | Hash-chained receipts with Ed25519 signing |
| `topology.py` | Tensor-native topology and bridge detection |
| `mpc.py` | Model-predictive structural control |
| `equivariant.py` | Permutation-equivariant GNN executive |
| `transactions.py` | Transactional graph mutations |
| `version.py` | Version constant (5.3.2) |

### Benchmark (`src/lgae_v3/benchmark/`)

| File | Purpose |
|------|---------|
| `tasks.py` | 7 synthetic tasks (A–G) with known-optimal actions |
| `baselines.py` | Random, spectral-heuristic, oracle controllers |
| `metrics.py` | Diagnosis accuracy, mutation regret |
| `harness.py` | Benchmark harness |
| `policy_qualification.py` | Deterministic policy qualification |
| `counterfactual.py` | Counterfactual dataset, Q-network training, evaluation |

### Scripts

| Script | Purpose |
|--------|---------|
| `qualify.py` | Geometry qualification (9 checks) |
| `qualify_production.py` | Production dynamics qualification (8 checks) |
| `qualify_policy.py` | Policy qualification (accuracy, regret) |
| `compare_baselines.py` | Compare learned vs random vs spectral vs oracle |
| `run_real_experiment.py` | Karate Club real-world experiment |
| `train_q_controller.py` | Train Q(S,a) on counterfactual outcomes |
| `scale_qualification.py` | Empirical complexity curves |
| `torch_compile_qualification.py` | Eager vs compiled performance |
| `generate_manifest.py` | Generate and verify SHA-256 manifest |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CHANGELOG.md` | Version history |
| `BUILD_REPORT.md` | Build and release report |
| `docs/COMPLETE_GUIDE.md` | This document |
| `docs/ARCHITECTURE.md` | Architecture overview |
| `docs/MATHEMATICS.md` | Mathematical foundations |
| `docs/AUDIT_RESPONSE.md` | Audit findings and responses |
| `docs/READING_LIST.md` | Recommended reading |
| `docs/V5_3_0_PRODUCTION_DYNAMICS.md` | v5.3.0 production dynamics notes |

### Release artifacts

| File | Purpose |
|------|---------|
| `MANIFEST.sha256.json` | SHA-256 manifest (126 files) |
| `release_verification.json` | Release verification evidence |
| `qualification_report.json` | Geometry qualification output |
| `production_qualification_report.json` | Production qualification output |
| `policy_qualification_report.json` | Policy qualification output |
| `example_output.txt` | CLI demo output |
| `dist/lgae_v3-5.3.2-py3-none-any.whl` | Distribution wheel |

---

## 15. Verification and Release

### Current release: v5.3.3

| Check | Result |
|-------|--------|
| Version constant | 5.3.3 |
| pyproject.toml | 5.3.3 |
| Schema constants | LGAE_GEOMETRY_V5_3_3 |
| README | LGAE v5.3.3 |
| CHANGELOG | v5.3.3 |
| BUILD_REPORT | v5.3.3 |
| Tests | 652 passing |
| Geometry qualification | PASS (9/9) |
| Production qualification | PASS (8/8) |
| Policy qualification | PASS (100% accuracy, 0.0 regret) |
| Wheel SHA-256 | 73aed41bc94ca0c3e5c5a77c2afc2e0f165334c46cbc914687d2bbcd78cbae58 |
| Manifest | 132 files verified |

### Verification procedure

1. Build the wheel: `python -m build --wheel`
2. Install the wheel: `pip install --force-reinstall --no-deps dist/*.whl`
3. Run tests from wheel (no PYTHONPATH): `python -m pytest -q`
4. Run qualification scripts from installed package
5. Regenerate reports
6. Regenerate manifest as the **final step**
7. Verify manifest: `python scripts/generate_manifest.py --check`

### CI

GitHub Actions workflow in `.github/workflows/ci.yml` runs the full
test suite on push/PR. Badge in README links to
`dawsonblock/LGAE` repository.

---

## 16. Version History

### v5.3.3 — Reproducibility repair (current)

- Canonical action ordering (ACTION_ORDER, ACTION_TO_INDEX)
- Removed Python hash() from deterministic logic
- DeterministicRNGContext with domain-separated substreams
- ReproducibilityInfo and qualification_id in all reports
- All 652 tests pass under PYTHONHASHSEED=0,1,2,42,123456
- All qualification reports byte-for-byte identical across runs
- Policy qualification deterministic at 100% accuracy, 0.0 regret

### v5.3.2 — Research improvements and safety architecture

**Research improvements:**
- Q(S,a) counterfactual controller (86% held-out accuracy)
- GraphFeatureBaseline replaces GraphHashBaseline
- Hierarchical candidate retrieval (top-64 + KNN, 512 max pairs)
- Dynamic gauge generator norm clamping
- Equilibrium barrier upgraded to dynamics residual
- Counterfactual dataset across 8 topology families
- Permutation-equivariant GNN executive
- Model-predictive structural control (MPC)

**Safety architecture:**
- ProductionConfig vs ResearchConfig
- Governor certification levels (CERTIFIED_GLOBAL / SAMPLED_LOCAL / HEURISTIC_PROXY)
- Mutation authority levels (REVERSIBLE / STRUCTURAL / IRREVERSIBLE)
- Checkpoint Merkle root
- Ed25519 receipt signing
- Bayesian curvature hysteresis (Normal-Inverse-Gamma)
- Tensor-native topology and bridge detection
- Scale qualification script
- torch.compile performance qualification (honest negative result)
- Task G: information gain active experimentation

**Tests: 652 (was 629 in v5.3.2, was 573 in v5.3.1)

### v5.3.1 — Integrity and baseline-comparison fixes

- Fixed nondeterministic policy qualification (seed before init)
- Removed circular benchmark utility in Task A
- Added baseline comparison (random, spectral, oracle)
- Added truly held-out structurally different task variants
- Added Karate Club real-world experiment (negative result)
- Fixed version identity mixing
- Rebuilt wheel from installed package
- Regenerated manifest as final step

**Tests:** 573

### v5.3.0 — Production dynamics hardening

- Curvature EMA and variance tracking
- Latent equilibrium barrier
- Graph hash baseline for credit assignment
- Transactional mutations with quarantine
- Hash-chained mutation receipts
- Safe checkpoint format (safetensors)
- Multi-horizon shadow certification
- Betti-number and bridge protection

### v5.2.0 — Policy qualification

- Structural executive with action heads
- Synthetic benchmark tasks (A–F)
- Policy qualification harness
- Deterministic evaluation

### v5.1.1 — Closed-loop hardening

- Reverse-edge inverse consistency for dynamic gauge
- ANN neighbor index
- Causal edge semantics
- Hypergraph support

### v5.0.0 — Initial release

- Sheaf Laplacian diffusion with SO(d) gauge
- Ollivier/Bakry-Émery/LLY curvature operators
- Spectral certification (exact + LOBPCG)
- Persistent homology signatures
- Fiber management (spawn/prune)
- Geometry governor with shadow evaluation

---

*This document is maintained as part of the LGAE repository.
For the latest version, see [dawsonblock/LGAE](https://github.com/dawsonblock/LGAE).*
