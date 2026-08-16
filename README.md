<div align="center">

# LGAE-v5.3.0 / 1LR: Governed Adaptive Geometry Engine

**A Multi-Timescale Geometric Controller for Self-Evolving Graph and Fiber-Bundle Latent Spaces**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![CI Status](https://github.com/dawsonblock/1LR/actions/workflows/ci.yml/badge.svg)](https://github.com/dawsonblock/1LR/actions)
[![Tests](https://img.shields.io/badge/tests-573%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gauge: SO(d)](https://img.shields.io/badge/gauge-SO(d)%20Invariance-purple.svg)]()
[![Validation: synthetic-only](https://img.shields.io/badge/validation-synthetic%20only-orange.svg)]()

</div>

---

## Overview

**LGAE-v5.3.0 (`1LR`)** is a research-grade governed geometric learning engine and structural controller. It operates over graph-structured data and continuous fiber bundles, combining continuous field diffusion, Lie-algebra gauge connections, discrete Ricci-flow surgery, and multi-operator curvature diagnostics.

> **Validation status (read first).** The geometry/numerical oracles (Bakry–Émery, LLY, Ollivier, SO(d), log-Sinkhorn, LOBPCG) are verified against analytic ground truth and are the strongest part of this codebase. The *structural-policy* qualification (diagnosis accuracy / mutation regret) is measured on **six hand-authored synthetic tasks** with no external dataset, no baseline controller for comparison, and "held-out" seeds that reuse the same task structures. It demonstrates internal consistency of the training loop, **not** real-world structural-diagnosis generalization or deployment safety. See [Validation boundaries](#validation-boundaries) below.

### v5.3.0 — Production Dynamics Hardening

v5.3.0 hardens the long-running dynamics around the v5.2 policy-qualified controller: exact native `SO(d)` gauge transport is retained, external sheaf maps gain non-expansive transport guards, automatic surgery gains curvature EMA/variance hysteresis, directed kernels gain stationary-measure Γ₂ symmetrization, ANN/neighbor caches become transaction-generation aware, slow structural updates can wait for latent equilibrium, and mutation learning uses graph-conditioned counterfactual advantages.

Release gates: **573 tests passing**, geometry qualification **9/9**, production-dynamics qualification **8/8**, and structural-policy qualification **PASS** at **83.3%** synthetic diagnosis accuracy with **0.0176** mean regret (deterministic; previously the gate was nondeterministic, see `CHANGELOG`). See `docs/V5_3_0_PRODUCTION_DYNAMICS.md`.

### v5.2.0 — Structural Policy Qualification

v5.2.0 qualifies the learned structural proposal layer in addition to the geometry governor. It adds compact latent-state diagnostics, bounded learned mutation magnitudes, explicit risk learning from REJECT/QUARANTINE, a measured ensemble-variance information-gain proxy, persistent pending credit, and a supervised policy-prior head for controlled qualification.

Release policy gate (`scripts/qualify_policy.py`): structural-diagnosis accuracy and mutation regret across five held-out seeds after training on 864 synthetic structural outcomes. **The benchmark is controlled/synthetic** — the "held-out" seeds reuse the same six task structures with different latent noise, and each task's utility is constructed so the labeled "correct" action is the argmax. This verifies the training loop is internally consistent; it does **not** establish real-world structural-diagnosis generalization, governor certification, or deployment safety.

### v5.1.1 — Closed-Loop Authority Integration

This release hardens the v5 learning layer and re-verifies the v3/v4 geometry substrate. The authoritative loop is now:

```text
observe → learned proposal/target → counterfactual comparison → uncertainty gate
→ LGAEEngine transaction → ACCEPT / QUARANTINE / REJECT
→ outcome receipt → long-horizon credit → executive/ensemble update
```

Key corrections:

- `LGAEEngine` is the sole structural commit authority; governor-only use is read-only certification.
- `QUARANTINE` is never treated as execution and never receives committed-action credit.
- graph, fiber, and gauge changes all have transactional shadow/rollback/quarantine paths.
- safe checkpoints persist complete graph/fiber/gauge quarantine shadows and use the canonical v5.1.1 schemas.
- bootstrap ensemble uncertainty is read-only with respect to the authoritative executive and updates online from observed outcomes.
- conformal residual calibration uses the correct finite-sample order statistic and is used by the uncertainty gate.
- long-horizon credit advances on every loop step and finalized outcomes feed back into executive/ensemble learning.
- the executive observes actual active-fiber capacity and includes learned node/edge target scorers.
- sheaf-Laplacian diffusion sign and isolated-node adjacency behavior are corrected.
- persistent-homology bottleneck distance uses a true minimax matching decision procedure rather than minimum-sum Hungarian assignment.
- Bakry–Émery/CDE local certification again requires the full two-hop neighborhood; incomplete local neighborhoods fail uncertain rather than certifying a truncated problem.
- ANN neighbor search implements the common index protocol, removes self-neighbors before truncation, uses `-1/+∞` padding, and labels the NumPy fallback accurately.
- dynamic-gauge reverse pairs are constructed with antisymmetric generators so `U_ji = U_ij^T`.
- causal-edge reclassification clears stale cached causal parent/child structure; hyperedges validate endpoints and weights at insertion.

The older SO(d), log-Sinkhorn, reversible Γ₂, log-conformal Ricci-flow, cooldown/hysteresis, sparse LOBPCG, transactional dry-run, and fixed-capacity `torch.compile` boundaries remain release gates rather than being replaced.

### v5.1.0 — Advanced Structural Modules (historical)

Six modules introduced in v5.1.0, with the authoritative subset hardened/integrated in v5.1.1. They extend the structural learning loop with richer geometry, causality, and higher-order relationships:

1. **Dynamic gauge connections** (`dynamic_gauge.py`): Context-conditioned SO(d) transport where `U_ij = exp(skew(f_θ(z_i, z_j, c_t)))`. Connection matrices adapt to latent states and task context, enabling richer message passing than static edges. Includes `DynamicGaugeBank`, `StaticGaugeAdapter` for backward compatibility, and `gauge_alignment_loss`.

2. **Multi-timescale adaptation** (`timescales.py`): Separates adaptation into fast (every step: gauge, gates), medium (every ~100: affinity, fibers), and slow (every ~1000: length, topology) timescales. Prevents mutual drift between representation and structure learning. Enforces minimum convergence before slower timescales activate.

3. **Sheaf-adjacency diffusion** (`sheaf_diffusion.py`): Replaces pure Laplacian diffusion with sheaf-adjacency form plus normalization and gating. Avoids suppressing useful disagreement signals at depth. Includes `agreement_gate` for selective message passing and `compare_diffusion_methods` for empirical selection.

4. **ANN-backed neighbor index** (`ann_index.py`): Approximate nearest neighbor backend with FAISS or dependency-free random-projection ANN fallback. Pipeline: latent Z → ANN index → 96 candidates → exact reranking → 32 final neighbors. Includes `measure_recall` for Recall@k qualification and index refresh policies.

5. **Causal edge semantics** (`causal_edges.py`): Distinguishes association edges from causal edges. Supports causal-edge bookkeeping, intervention-propagation heuristics, counterfactual graph traversal, and causal path analysis. Temporal direction scoring remains an experimental heuristic rather than a full structural-causal-model or Granger implementation.

6. **Hypergraph / higher-order relationships** (`hypergraph.py`): Extends the graph to hyperedges connecting 3+ nodes. Captures relationships that cannot be decomposed into pairwise interactions. Includes clique expansion and star expansion for spectral analysis, and hypergraph Laplacian diffusion.

### v5.0.0 — Structural Learning Loop

The architectural jump from "geometry detects → rules propose → governor validates" to:

```
geometry observes → learned executive predicts → counterfactuals compete → governor certifies → outcomes train the executive
```

**Five priority additions:**

1. **Learned structural executive** (`executive.py`): A proposal model that observes local geometry, task residuals, uncertainty, capacity, edge role, and recent mutation history, then scores structural actions (NO_OP, ADD_EDGE, PRUNE_EDGE, REWEIGHT_AFFINITY, REWEIGHT_LENGTH, SPAWN_FIBER, PRUNE_FIBER, CHANGE_GAUGE, COUPLED_REWEIGHT). Objective: `m* = argmax[E[ΔU(m)] + ν·IG(m) - λ·C(m) - μ·R(m)]`. Proposal generator only; governor remains authority.

2. **Long-term mutation credit assignment** (`credit.py`): Tracks mutation receipts and outcomes at horizons {16, 100, 1000}. Discounted return: `R = Σ γ^τ ΔU_{t+τ}`. The executive learns from its own structural history by comparing initial predictions to long-term outcomes.

3. **Calibrated uncertainty** (`uncertainty.py`): Ensemble-based epistemic uncertainty `p(ΔU|m,S)`. LCB acceptance gate: `LCB(m) = E[ΔU_m] - β·σ_m`. Only positive LCB → auto-accept; uncertain but interesting → QUARANTINE. Conformal calibration with coverage guarantees.

4. **Stability/plasticity + consolidation** (`consolidation.py`): Capacity budget `B_t = Σ d_i + α|E|`. Growth justification `ΔU/ΔB > τ_efficiency`. Fiber lifecycle: NEW → PROBATION → MATURE → PROTECTED / UNUSED → PRUNE. Probation gate `g(t)` slowly increases, giving new fibers time to integrate before evaluation.

5. **Task-grounded benchmark harness** (`benchmark/`): 6 synthetic tasks with known-optimal structural changes:
   - Task A: Long-range bottleneck → ADD_EDGE
   - Task B: Local representational complexity → SPAWN_FIBER
   - Task C: Noisy spurious edge → PRUNE_EDGE
   - Task D: Coordinate-frame mismatch → CHANGE_GAUGE
   - Task E: Distribution shift → SPAWN_FIBER
   - Task F: Nothing wrong → NO_OP

   Metrics: structural diagnosis accuracy and mutation regret `R_t = U(m_t*) - U(m_t)`.

**Structural counterfactual engine** (`counterfactual.py`): Compares multiple candidate actions from the same state. NO_OP is always included as baseline. If no candidate beats NO_OP after accounting for risk and cost, the system does nothing.

**Closed loop** (`structural_loop.py`): Ties everything together — observe, predict, counterfactual, certify, execute, train.

### v4.1.3 — Deep Audit: Sparse Scaling, Float64 Discrepancy, ANN Index

- **Analytic vertex selection policy**: BE/CDE vertices now selected as union of highest transport pressure, lowest LLY, highest operator discrepancy, and mutation-touched nodes. Replaces the old `order[:bakry_nodes]` heuristic.
- **Local neighborhood cap**: `local_dense_diagnostic` now caps at `max_local_nodes=256` with `radius=1` to prevent O(N) local matrices on dense k-NN graphs. N=2500 audit drops from 73s to 3.2s.
- **Float64 discrepancy validation**: Sparse discrepancy verified against dense reference to 1e-10 tolerance in float64 across 16 parameterized cases.
- **Deliberate duplicate edge tests**: COO coalescing verified with deliberately duplicated edges in both operators.
- **v4 checkpoint length mandatory**: Safe checkpoints now require `length` tensor for v4+ state. Legacy migration path infers `ℓ=1/a` only for schema < 4.
- **Stale quarantine detection**: Quarantine entries persist `base_graph_hash`; after restart, acceptance checks `H(G_current) == H(G_base)` and rejects stale quarantines.
- **Parameterized governance hash**: Every decision-affecting config field tested for hash sensitivity (90+ parameterized cases).
- **StructuralMutation protocol**: All mutation classes implement `touched_region()` returning affected node indices. Enables unified multi-horizon certification for graph and fiber mutations.
- **Neighbor index abstraction**: `NeighborIndex` protocol with `ExactChunkedKNN` reference backend. `build_knn_graph()` and `recall_at_k()` for ANN backend validation. Pluggable architecture for future HNSW/FAISS backends.
- **Forman reference tests**: K2, weighted path, weighted star, uniform reduction, and tree curvature verified against analytic formulas.
- **Multi-horizon decision combinations**: All 9 decision aggregation patterns tested (AAA→A, AAQ→Q, AQA→Q, QQA→Q, AAR→R, QRA→R, RRR→R, early REJECT, QUARANTINE+REJECT).

### v4.1.2 — Geometry-Mode Tiers, Mutation Split, PH Bottleneck

- **Geometry-mode tier separation**: `candidate_geometry_mode`, `audit_geometry_mode`, `certificate_geometry_mode` allow independent configuration of the candidate proxy, audit, and certificate tiers. Empty values fall back to `curvature_weight_mode`.
- **Metric-measure mutation split**: `ReweightAffinity`, `ReweightLength`, and `CoupledReweight` provide explicit control over which field (affinity, length, or both with coupling policy) a mutation affects. The legacy `ReweightEdge` remains for backward compatibility.
- **PH bottleneck distance**: `persistent_homology_bottleneck_drift()` computes the proper bottleneck distance between persistence diagrams using Hungarian matching, replacing the summary-statistic drift when `use_bottleneck_ph_drift=True`.
- **Multi-horizon fiber mutations**: `evaluate_latent_transition()` now applies the same max-severity multi-horizon certification as graph mutations.
- **65 adversarial/numerical tests**: NaN/Inf/zero injection across affinity, length, latent, gauge, curvature, and transition probabilities. Disconnected graphs, bridge pruning, extreme configs. All fail closed.

### v4.1.1 — Sparse Governance Integrity

This release fixes the integration defects identified in the forensic audit of v4.1.0:

- **Sparse governor operational end-to-end**: `SparseDualOperatorState` now provides `p_diagnostic`/`p_actuation` properties and a `local_dense_diagnostic()` method for local BE/CDE extraction. The governor no longer crashes for N>2048. BE/CDE audits use local 2-hop neighborhood extraction instead of global dense matrices.
- **Sparse discrepancy coalesces duplicates**: Uses `torch.sparse_coo_tensor.coalesce()` to correctly accumulate duplicate directed edges from mutual k-NN symmetrization. Sparse discrepancy now matches dense reference exactly.
- **Safe checkpoints persist metric length**: The `length` tensor is now stored in safetensors alongside `weight`, preserving the metric-measure separation through save/restore.
- **Durable quarantine**: Safe checkpoints now persist complete shadow graph tensors, allowing quarantined transactions to be resumed after reload.
- **Governance hash includes v4.1 fields**: `shadow_horizons`, `ricci_flow_target`, `ricci_flow_coupled` are now included in the governance fingerprint.
- **Mutation serialization complete**: `RicciFlowReweight.target_field` and `.coupled` are now serialized in mutation specs, with backward-compatible defaults for pre-v4.1 records.
- **Multi-horizon max severity**: The final decision is now `max(severity)` across all horizons — any QUARANTINE propagates, not just REJECT.
- **Metric-measure Forman**: `weighted_forman_edge` now uses the canonical formula with vertex measure m₁, edge measure m₂ (affinity), and metric ω (length), instead of the old square-root weight ratio formula.
- **Version identity unified**: Single `version.py` module provides `VERSION` used by package, CLI, checkpoints, receipts, manifest, and qualification.

### v4.1 — Metric–Measure Separation + Multi-Horizon Certification

This release closes the metric/affinity semantic gap identified in the v4.0 review:

- **Metric–measure separation**: The graph state now carries two independent edge scalars: `weight` = **affinity** (conductance, transition probability) and `length` = **metric length** (geometric distance). The Markov operator uses `P_uv = a_uv / Σ a_uj` (affinity), while shortest-path distances and curvature ground costs use `d_ℓ(x,y) = shortestpath_ℓ(x,y)` (length). This gives the system a coherent `(V, d_ℓ, P_a, m)` metric-measure structure instead of overloading one scalar.
- **Weighted ORC**: ground cost from `length`, lazy measures from `P(affinity)` — cleanly separating "how far apart are states?" from "how likely is information to move?"
- **Weighted LLY**: Lipschitz constraint from `length`, Laplacian from `P(affinity)` — matching the Bai–Huang–Lu–Yau formulation where metric `d` and transition rule `P` are independent.
- **Literature-faithful weighted Forman**: `weighted_forman_edge` implements the canonical formula with explicit square-root weight ratios. The old degree-substitution heuristic is retained as `weighted_af3_proxy` (clearly labeled, not claiming canonical status).
- **Ricci flow target selection**: `ricci_flow_target` config selects whether Ricci flow modifies `length` (geometrically canonical) or `weight` (affinity), with optional coupling.
- **Multi-horizon shadow certification**: `shadow_horizons = [1, 2, 4, 8, 16]` requires a mutation to remain admissible across ALL horizons, not just one rollout length.
- **Scalability claims corrected**: sparse operator storage is O(Nk), but k-NN construction is O(N²D) compute with bounded memory via chunking. Documented as "bounded-memory exact k-NN", not sub-quadratic ANN.

### v4.0 — Sparse Weighted Geometry

This release closes the scalability and weighted-geometry gaps identified in the v3.3 audit:

- **Sparse dual operators**: `SparseDualOperatorState` replaces the dense `N×N` actuation and diagnostic diffusion operators with `O(Nk)` edge-list representations. The diagnostic diffusion uses k-NN without materializing the full pairwise distance matrix. Operator discrepancy is computed on the union of supports.
- **Weighted curvature backends**: `curvature_weight_mode='weighted'` is now supported. Weighted Ollivier uses edge-weight-proportional lazy measures and Dijkstra shortest-path costs. Weighted LLY uses the weighted normalized Laplacian with shortest-path-distance boundary conditions. Weighted AF3 uses weighted degree instead of unweighted degree.

### v3.3 — Authority and Persistence Hardening

This release closes the state-authority gap identified in the v3.2 audit:

- **Canonical authority hash** `H(G, g_e, U, F, C_g)` binds graph, gauge, fiber, and governance config into a single SHA-256 commitment.
- **Slot-generation cryptographic binding**: `slot_generation` is now included in the graph state hash, preventing ABA-style slot reuse from going undetected.
- **Graph/gauge generation synchronization**: the graph is the canonical generation authority; gauge bank generations sync from the graph at init, commit, and checkpoint boundaries.
- **Checkpoint config enforcement**: structural config mismatch fails immediately; governance mismatch requires explicit `allow_governance_mismatch=True` migration flag.
- **Optimizer checkpoint semantics**: `optimizer_load_policy` supports `"restore"`, `"reset"`, and `"reject"` — no more silent mixing of checkpoint parameters with stale optimizer history.
- **Safe checkpoint format**: `safetensors + JSON` directory format for untrusted interchange (no pickle deserialization).
- **Optimizer-generic slot reset**: clears all tensor-valued optimizer state matching edge capacity, not just Adam-specific keys (handles Adagrad, RMSProp, etc.).
- **Hash-chained receipts**: tamper-evident ledger with `H_i = SHA256(H_{i-1} || R_i)` and `verify_receipt_chain()`.
- **Receipts bind gauge authority**: accepted-mutation receipts now include `base_gauge_hash` and `authority_hash_after`.
- **Exact manifest coverage**: `scripts/generate_manifest.py` with `--check` mode; `.gitignore` explicitly declared as excluded.

### Core Architecture Principle
> **Field dynamics are sparse and compiled; discrete evolution is transactional and eager; curvature diagnoses rather than directly dictates topology.**

```
       +-------------------------------------------------------------+
       |                  Continuous Field Dynamics                  |
       |  - Latent states z in R^{N x D} with dynamic fiber gating   |
       |  - Sparse row-stochastic Markov diffusion (O(E x D))        |
       |  - SO(d) gauge parallel transport: U_e in SO(d_g)           |
       +------------------------------+------------------------------+
                                      |
                         Fast Geometric Signals (Gamma, r, Var)
                                      v
       +-------------------------------------------------------------+
       |               Transaction & Shadow Evaluation               |
       |  - Eager shadow rollout with dual actuation/diagnostics     |
       |  - Log-conformal Ricci flow: w' = clamp(w * exp(-dt * dk))  |
       |  - Graph surgery: add, reweight, prune with hysteresis      |
       +------------------------------+------------------------------+
                                      |
                      Curvature & Topological Audits
                                      v
       +-------------------------------------------------------------+
       |                   Authoritative Governor                    |
       |  - Exact LLY & Log-Sinkhorn Ollivier (W1 optimal transport) |
       |  - Reversible Bakry-Émery CD(K, N) with Schur complements   |
       |  - Sparse LOBPCG spectral certificate & beta_0 protection   |
       |  - Accept, Reject, or Quarantine with SHA-256 state locks   |
       +-------------------------------------------------------------+
```

---

## Key Features & Hardening

### 1. $\mathrm{SO}(d)$ Gauge Connection Bank (`SOConnectionBank`)
* **Lie-Algebra Parameterization**: Generator parameters live in unconstrained space $R_e$, strictly mapped through the skew-symmetric algebra $\mathfrak{so}(d)$ via $A_e = \frac{1}{2}(R_e - R_e^T)$ and mapped to $\mathrm{SO}(d)$ via Cayley retraction or Matrix Exponential ($\exp(A_e)$).
* **Guaranteed Invariance**: Connections strictly satisfy $U_e^T U_e = I$ and $\det(U_e) = +1$ to machine precision across arbitrary Euclidean optimizer steps.
* **Slot Generation Lifecycle ($g_e$)**: Monotonic generation counters $(g_e \leftarrow g_e + 1)$ track slot allocation and retirement. Generations are cryptographically committed in the graph state hash and synchronized between graph and gauge authorities.
* **Optimizer Momentum Isolation**: When an edge slot is retired or reused, all tensor-valued optimizer state slices whose leading dimension matches edge capacity are zeroed (optimizer-generic: handles Adam, AdamW, SGD, Adagrad, RMSProp, etc.). Scalar state (step counters) is preserved.

### 2. Stable Optimal Transport (Log-Sinkhorn Ollivier Curvature)
* **Log-Domain Scaling**: Eliminates probability-space underflow at small entropic regularization $\epsilon$.
* **Zero-Mass Pruning**: Exact support removal for unvisited states.
* **Marginal-Residual Certification**: Convergence validated against recovered coupling marginals rather than dual scaling differences alone.
* **Exact Ground-Truth Oracle**: High-precision linear programming (`exact_lp`) retained for qualification checks.

### 3. Reversible $\Gamma$-Calculus & Bakry–Émery ($CD(K, N)$)
* **Continuous-Time Reversible Markov Generators**: $\Delta = P - I$ formed with detailed-balance volume measure reconstruction.
* **Float64 Conditioning**: Precision row re-normalization and diagonal ULP cancellation.
* **$\Gamma$-Nullspace Schur Complement**: Eliminates uncoupled higher-hop coordinates ($B_{\text{eff}} = B_{pp} - B_{pn} B_{nn}^+ B_{np}$), preventing false-positive curvature anomalies.

### 4. Log-Conformal Ricci Flow & Surgery Hysteresis
* **Weight Positivity**: Multiplicative updates $w \leftarrow \text{clamp}(w \cdot \exp(-\Delta t(\kappa - \kappa^*)), w_{\min}, w_{\max})$ guarantee weights never cross zero.
* **Anti-Thrashing Cooldown**: Canonical edge cooldown tracker separates addition, deadband, and pruning regions.
* **$O(V+E)$ Bridge Filter**: Rejects disconnecting edge removals before triggering expensive global audits.

### 5. `torch.compile` Compatibility & Predictability
* **Fixed-Shape Buffer Bucketing**: `GraphBuffers` round capacity to fixed-size buckets with in-place value refresh (`refresh_padded_markov_edges_`).
* **Dormant Fiber Channel Suppression**: Inactive latent coordinates are zeroed post-diffusion to prevent hidden energy accumulation.

---

## Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.6+
- NumPy, SciPy, NetworkX, PyYAML

```bash
# Clone the repository
git clone https://github.com/dawsonblock/1LR.git
cd 1LR

# Install in editable mode with development dependencies
python -m pip install -e '.[dev]' --no-build-isolation
```

> **Naming.** The release is **LGAE-v5.3.0** (this README) and the repo is **`1LR`**. The Python distribution/module is `lgae-v3` / `lgae_v3` — a historical name kept for import stability across releases (renaming would touch every import in 53 source + 31 test files for no functional gain). So: `pip install lgae-v3` gives you `import lgae_v3` at version `5.3.0`. The CLI is `lgae-v3`. Older docs may say "LGAE-v3.2"; that refers to the v3.2-era architecture, not the current version number.

---

## Quickstart & Code Examples

### 1. Gauge Parallel Transport on Fiber Bundles

```python
import torch
from lgae_v3 import LGAEConfig, LGAEEngine, make_bucketed_graph_buffers

# Configure fiber dimensions and gauge group
cfg = LGAEConfig()
cfg.fiber.d_base = 8
cfg.fiber.d_max = 16
cfg.fiber.gauge_dim = 8
cfg.fiber.gauge_parameterization = "cayley"  # 'cayley' or 'exp'

# Initialize bucketed graph buffers
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
graph = make_bucketed_graph_buffers(num_nodes=4, edges=edges, bucket_size=256)

# Create engine and execute gauge-covariant diffusion
engine = LGAEEngine(graph, cfg)
z_next = engine.diffuse_(eta=0.01)

# Verify SO(d) invariants
orth_err, det_err = engine.gauge_connections.invariant_error()
print(f"Max orthogonality error: {orth_err.max():.2e}")
print(f"Max determinant error:   {det_err.max():.2e}")
```

### 2. Differentiable Training Core with Optimizer Isolation

```python
import torch
from torch import nn
from lgae_v3 import LGAEConfig, LGAEEngine, LGAETrainCore
from lgae_v3.training import padded_markov_edges_with_slots, train_step

cfg = LGAEConfig()
cfg.fiber.d_base = 4
cfg.fiber.d_max = 8
cfg.fiber.gauge_dim = 4

graph = make_bucketed_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], bucket_size=32)
engine = LGAEEngine(graph, cfg)

# Setup train core with shared gauge connections
decoder = nn.Linear(8, 2)
core = LGAETrainCore(engine.fibers, decoder, gauge_bank=engine.gauge_connections, gauge_dim=4)
optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)

# Padded fixed-shape buffers for torch.compile stability
src, dst, w, valid, slot, reverse = padded_markov_edges_with_slots(graph, max_edges=32)
target = torch.randn(4, 2)
pressure = torch.zeros(4)

# Execute one step (automatically registers optimizer for slot lifecycle management)
metrics = train_step(
    core, engine, optimizer,
    target=target, src=src, dst=dst, weight=w, valid=valid,
    bottleneck_pressure=pressure, edge_slot=slot, reverse=reverse,
    step=0, spawn_interval=50
)
print("Step Loss:", metrics["loss"].item())
```

### 3. Curvature Auditing & Governed Surgery

```python
from lgae_v3.mutations import AddEdge, PruneEdge, ReweightEdge

# Propose an edge addition
mutation = engine.propose_midpoint_edge()

# Shadow-evaluate and govern transaction
result = engine.evaluate_and_maybe_commit(mutation)
print("Decision:", result.decision.value)  # 'accept', 'reject', or 'quarantine'
print("Reasons:", result.reasons)
print("Authority hash after:", result.metadata.get("authority_hash_after"))
```

### 4. Checkpoint Authority & Safe Persistence

```python
# Save in safe (safetensors + JSON) format for untrusted interchange
engine.save_checkpoint("checkpoint_dir/")

# Save in legacy pickle format (trusted local use only)
engine.save_checkpoint("checkpoint.pt")

# Load with config authority enforcement
engine2.load_checkpoint_("checkpoint_dir/")

# Load with explicit governance migration
engine2.load_checkpoint_(
    "checkpoint_dir/",
    allow_governance_mismatch=True,
    optimizer_load_policy="restore",  # "restore" | "reset" | "reject"
)

# Verify canonical authority hash
print("Authority:", engine2.authority_hash())
engine2.assert_generation_sync()  # raises on graph/gauge generation divergence
```

### 5. Hash-Chained Receipt Ledger

```python
from lgae_v3.receipts import mutation_receipt, append_receipt, verify_receipt_chain

# Create and append a chained receipt
receipt = mutation_receipt(
    result,
    authority_state_hash_before=engine.authority_hash(),
    gauge_authority_hash=engine.gauge_connections.state_hash(),
)
append_receipt("ledger.jsonl", receipt)

# Verify the entire chain is tamper-evident
is_valid, errors = verify_receipt_chain("ledger.jsonl")
assert is_valid
```

---

## CLI Utilities

```bash
# Run qualification suite across all geometric and numerical oracles
python scripts/qualify.py

# Run full test suite with zero warnings
pytest -v -W error

# Run self-evolving graph demo
lgae-v3 demo --nodes 10 --steps 4

# Cross-validate exact LLY curvature paths
lgae-v3 qualify-lly --graph cycle --nodes 6

# Generate or verify the SHA-256 integrity manifest
python scripts/generate_manifest.py           # write manifest
python scripts/generate_manifest.py --check   # verify manifest
```

---

## Mathematical Oracles & Qualification Matrix

| Metric / Oracle | Graph / Test Case | Theoretical Target | LGAE-v5.3 Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Bakry–Émery $K_\infty$** | Path $P_4$ (interior) | $1 - \frac{\sqrt{2}}{2} \approx 0.292893$ | `0.2928932188` | **PASS** |
| **Bakry–Émery $K_\infty$** | Path $P_4$ (endpoints) | $1.0$ | `1.0000000000` | **PASS** |
| **Bakry–Émery $K_\infty$** | Complete $K_2$ | $2.0$ | `2.0000000000` | **PASS** |
| **Exact LLY Agreement** | $K_2, P_4, C_4, K_3$ | $\kappa_{\text{LP}} = 2\kappa_{1/2}$ | Max error: `0.0` | **PASS** |
| **Weak Entropic Curvature** | $K_3$ (empty 2-hop shell) | $+\infty$ | `Infinity` | **PASS** |
| **Log-Sinkhorn vs LP** | Large metric / small $\epsilon$ | $998.0$ | `997.999999999` | **PASS** |
| **$SO(d)$ Invariance** | Post-Adam steps | $\|U^T U - I\|_F < 10^{-10}$ | `Pass` | **PASS** |
| **Sparse LOBPCG Spectral Gap** | Cycle $C_{24}$ | Matches exact $\lambda_2$ | `0.03407417` | **PASS** |

---

## Validation boundaries

The oracles above are verified against analytic ground truth and are the strongest claim this codebase makes. The following are **not** claims, and the README has been edited to stop implying them:

- **No real-world generalization claim.** The structural-policy qualification (`scripts/qualify_policy.py`) uses six hand-authored synthetic tasks (A–F). "Held-out" seeds 101–105 reuse the *same task structures* as training seeds 0–15, differing only in latent noise; several tasks fix the latent entirely. Diagnosis accuracy on these seeds measures memorization of the task constructor's labels, not generalization to unseen structures.
- **Benchmark utility is constructed, not discovered.** Each task's `utility()` is written so the labeled "correct" action is the argmax of Δ-utility by construction (e.g. Task A rewards inter-cluster edges + spectral gap, and the correct action is "add an inter-cluster edge"). The learned executive is graded on whether it recovers the action the utility was built to reward. This is a training-loop consistency check, not evidence that the governor diagnoses real structure.
- **No baseline comparison.** There is no comparison to a random-action controller, a spectral heuristic, or a vanilla GNN. It is therefore unknown whether the curvature/gauge/fiber machinery improves outcomes over a simpler controller on any task.
- **No deployment-safety proof.** The governor's transactional shadow/rollback/quarantine and fail-closed numerical behavior are engineering safeguards, not a formal safety argument.
- **`torch.compile` not performance-qualified this release.** The fresh CPU Inductor smoke timed out during compilation in the packaging container; the compiled-kernel architecture is inherited from v5.2 and not newly measured. See `release_verification.json`.

Two scripts are provided to begin closing these gaps; see [Baseline comparison](#baseline-comparison) and [Real-world experiment](#real-world-experiment).

---

## Baseline comparison

`scripts/compare_baselines.py` runs four controllers — random, spectral-heuristic, learned, oracle — on the same synthetic tasks and reports diagnosis accuracy and mean regret for each, on both the in-distribution tasks and truly held-out structurally-distinct task variants.

```
controller             split                  accuracy     regret
random                 in_distribution          0.1000     2.6918
spectral_heuristic     in_distribution          0.4667     2.2669
learned                in_distribution          0.8333     0.0176
oracle                 in_distribution          1.0000     0.0000
random                 held_out_structure       0.1000     3.0092
spectral_heuristic     held_out_structure       0.6000     2.4000
learned                held_out_structure       0.3000     0.6175
oracle                 held_out_structure       1.0000     0.0000
```

**Reading.** The learned executive clearly beats random and the non-learned spectral heuristic in-distribution. On structurally held-out tasks it **loses to the spectral heuristic on diagnosis accuracy** (30% vs 60%) but has lower regret (0.62 vs 2.40), suggesting it defaults to safe NO_OP on unseen structures rather than misdiagnosing. This is the first honest generalization signal the benchmark has produced, and it indicates the learned policy does not yet transfer as well as a cheap non-learned rule.

---

## Real-world experiment

`scripts/run_real_experiment.py` is a small real-world sanity check using Zachary's Karate Club (a real 34-node social network with two ground-truth communities, shipped with NetworkX) and a real downstream task: recovering the two communities from a latent embedding via clustering.

```
condition                accuracy    lambda2    edges
raw                        0.7059     0.1323       78
random_add                 0.7353     0.1632       84
spectral_heuristic         0.7941     0.1249       84
lgae_governed              0.6765     0.1323       78
```

**Reading.** On this real task the LGAE governance loop does **not** improve community recovery over the raw baseline: the governor commits zero mutations on a graph this small (all proposals are rejected by the shadow audits), and the engine's fiber latent is a slightly worse representation for clustering than the raw spectral embedding. A simple spectral heuristic (adding edges between embedding-close non-adjacent nodes) does best. This is an honest negative result for the governance loop on a small real graph and is reported as such — it does not support a "production-ready controller" claim. It is the starting point for understanding *when* the governance machinery helps, which remains an open question.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml            # Multi-version (Py 3.11 & 3.12) GitHub Actions CI
├── configs/
│   └── default.yaml          # Default engine and audit configurations
├── docs/
│   ├── ARCHITECTURE.md       # Full system architecture and state split
│   ├── MATHEMATICS.md        # Complete mathematical formulations and proofs
│   ├── V32_HARDENING.md      # Detailed v3.2 stability and gauge hardening notes
│   └── READING_LIST.md       # Theoretical background and references
├── examples/
│   └── run_lgae_v3.py        # End-to-end execution example
├── scripts/
│   ├── benchmark_compile.py  # torch.compile benchmark (eager/static/dynamic)
│   ├── benchmark_memory.py   # Memory footprint and step latency profiling
│   └── qualify.py            # Geometric qualification suite
├── src/lgae_v3/
│   ├── core/                 # Compatibility layer and engine entrypoints
│   ├── curvature/            # Bakry-Émery, CDE', Entropic, Forman, LLY, Ollivier
│   ├── training/             # LGAETrainCore, padded buffers, and train loops
│   ├── compile_utils.py      # Torch compile utilities
│   ├── config.py             # Strongly typed dataclass configurations
│   ├── evolution.py          # Authoritative LGAEEngine
│   ├── fibers.py             # FixedWidthFiberLatent & SOConnectionBank
│   ├── governor.py           # GeometryGovernor & transition audits
│   ├── metrics.py            # Gauge-covariant sparse diffusion metrics
│   ├── mutations.py          # Log-conformal Ricci flow & graph surgeries
│   ├── operators.py          # Actuation & diagnostic Markov operators
│   ├── receipts.py           # Cryptographic receipt logging
│   └── topology.py           # NetworkX conversion, Betti numbers & PH
└── tests/                    # 27 test modules with 512 verified unit/regression tests
```

---

## License

This project is licensed under the [MIT License](LICENSE).
