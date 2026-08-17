# LGAE v5.4.0 Roadmap — Unified Governed Structural Intelligence Runtime

**Status:** Planning document
**Target:** v5.3.3 (reproducibility hotfix) → v5.4.0+ (governed runtime)
**Date:** 2026-08-17

---

## Objective

Turn LGAE from a strong geometric research engine with fragmented
controllers into a **reproducible, cryptographically verifiable, governed
structural decision runtime**.

The hard success criterion:

> **Useful structural learning + reproducible evidence + enforced governance**

If any one of those three is absent, the system is not production-grade.

---

## Target Architecture

```
                    ┌──────────────────────────────┐
                    │   Graph / Hypergraph State   │
                    │ nodes / edges / fibers / SOd │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Canonical State Encoder       │
                    │ equivariant graph encoder     │
                    │ topology + geometry features  │
                    └──────────────┬───────────────┘
                                   │
                        embeddings + uncertainty
                                   │
                                   ▼
          ┌──────────────────────────────────────────────┐
          │ Candidate Structural Action Generator        │
          │                                              │
          │ ADD / REMOVE / REWEIGHT / GAUGE / FIBER     │
          │ NOOP / causal / hypergraph actions           │
          └───────────────────┬──────────────────────────┘
                              │
                              ▼
             ┌────────────────────────────────────┐
             │ Unified Q(S,a,target) Evaluator    │
             │                                    │
             │ utility                            │
             │ information gain                   │
             │ structural cost                    │
             │ uncertainty                        │
             │ risk                               │
             └─────────────────┬──────────────────┘
                               │
                               ▼
                     Structural MPC
                counterfactual rollout H steps
                               │
                               ▼
                     action sequence ranking
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Geometry / Safety Governor      │
              │                                 │
              │ authority class                 │
              │ curvature                       │
              │ topology                        │
              │ spectral checks                 │
              │ PH                              │
              │ causal consistency              │
              │ uncertainty                     │
              └───────────────┬─────────────────┘
                              │
                ACCEPT / QUARANTINE / REJECT
                              │
                              ▼
                   transactional mutation
                              │
                              ▼
                    authoritative graph
                              │
                              ▼
                  signed mutation receipt
                              │
                              ▼
                   replay / audit ledger
```

**The architectural principle that must never be violated:**

> AI proposes. Governor authorizes. Runtime commits.
> Never train the policy to bypass or subsume the governor.

---

## Release Plan

The full scope is 18 milestones. This is a multi-quarter roadmap, not a
single release. Splitting into staged releases prevents months of work
in a single branch with no intermediate verification — exactly the kind
of unreproducible mess this plan is trying to prevent.

| Release | Milestones | Focus | Type |
|---------|-----------|-------|------|
| **v5.3.3** | 1-3 | Reproducibility, release evidence, checkpoint verification | Pure engineering |
| **v5.4.0** | 4-6 | Production config, mutation authority, evidence/certification | Governance engineering |
| **v5.5.0** | 7-10 | Equivariant encoder, unified Q, hierarchical candidates, MPC | Controller rewrite |
| **v5.6.0+** | 11-18 | Procedural env, offline training, generalization, real-world, durable ledger, shadow, online, production qualification | Research + deployment |

### Why split

- **v5.3.3** is deterministic engineering with deterministic outcomes.
  It can be shipped and verified quickly.
- **v5.4.0** touches the governor core but is still engineering.
- **v5.5.0** is the largest substantive change — the controller rewrite.
  It should not be entangled with governance changes.
- **v5.6.0+** includes research milestones with uncertain outcomes.
  If generalization fails, the engineering milestones still shipped value.

---

## Phase 0 — Freeze the Current Baseline

Before touching algorithms, create a known baseline.

### Actions

1. Create tag `archive/v5.3.2-received`
2. Record forensic baseline:
   - Git commit hash
   - Archive SHA-256
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - OS
   - CPU
   - GPU (if applicable)
   - pytest collection count
   - Dependency lock hash
3. Run and archive:
   - `pytest --collect-only -q`
   - `pytest -q`
   - `python scripts/qualify_policy.py`
   - `python scripts/qualify.py`
   - `python scripts/qualify_production.py`
4. Run multiple deterministic environments:
   - `PYTHONHASHSEED=0`
   - `PYTHONHASHSEED=1`
   - `PYTHONHASHSEED=2`
   - `PYTHONHASHSEED=42`
   - `PYTHONHASHSEED=123456`
5. Record all results in `qualification/baseline_v5_3_2.json`

### Acceptance gate

- Current failures documented
- Current qualification inconsistencies documented
- No baseline artifacts overwritten
- Archive hash preserved

---

## Milestone 1 — Reproducibility Repair (v5.3.3)

**Do not touch learning architecture until reproducibility is fixed.**

### 1.1 Canonical action ordering

Create one authoritative action ordering:

```python
ACTION_ORDER = (
    NO_OP,
    ADD_EDGE,
    REMOVE_EDGE,
    REWEIGHT_EDGE,
    SPAWN_FIBER,
    REMOVE_FIBER,
    CHANGE_GAUGE,
    ADD_CAUSAL_EDGE,
    REMOVE_CAUSAL_EDGE,
)

ACTION_TO_INDEX = {
    action: idx
    for idx, action in enumerate(ACTION_ORDER)
}
```

Never derive semantic order from:
- sets
- dictionaries unless explicitly ordered
- filesystem traversal
- Python hash order

Replace `next(iter(correct_actions))` with
`min(correct_actions, key=ACTION_TO_INDEX.__getitem__)`.

### 1.2 Remove Python hash() from deterministic logic

Search globally for: `hash(`, `set(`, `frozenset(`, `next(iter(`,
`list(set`, `dict.keys`, `os.listdir`, `Path.iterdir`, `glob`.

Use deterministic hashing for persistent identity:

```python
import hashlib

def stable_u64(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
```

Use for: counterfactual family IDs, graph IDs, seed derivation,
benchmark identities, persistent cache keys.

### 1.3 Deterministic PRNG ownership

Create `DeterministicRNGContext` containing:
- `random.Random`
- `numpy.random.Generator`
- `torch.Generator` (CPU)
- `torch.Generator` (CUDA, if available)

Derive substreams with domain separation:

```
master_seed
   │
   ├── graph_generation
   ├── target_sampling
   ├── action_sampling
   ├── model_initialization
   ├── counterfactuals
   └── qualification
```

`s_i = H(s_master ∥ namespace)`

This prevents one new random call from changing every subsequent result.

### 1.4 Deterministic qualification

Every qualification report must contain:

```json
{
  "seed": 1234,
  "python_hash_seed": "0",
  "torch_deterministic": true,
  "source_commit": "...",
  "source_tree_sha256": "...",
  "python_version": "...",
  "torch_version": "...",
  "cuda_version": "...",
  "device": "...",
  "config_hash": "...",
  "dataset_hash": "..."
}
```

Run the same qualification twice. Compare canonical JSON byte-for-byte.

**Target:** Run A result digest == Run B result digest

---

## Milestone 2 — Release Evidence Rebuild (v5.3.3)

### 2.1 Generate reports, don't hand-edit them

Create `scripts/build_release_evidence.py` that automatically derives:
- pytest test count, pass/fail/skip count
- policy metrics
- geometry metrics
- benchmark metrics
- wheel hash
- source hash
- manifest hash
- dependency lock hash

Then generate `BUILD_REPORT.md`, `release_verification.json`, and
`QUALIFICATION.md` from one machine-readable evidence object.

README should not contain manually maintained qualification numbers.
Inject them during the release build.

### 2.2 Qualification run identity

Every complete qualification run gets an immutable identifier:

```
QID = SHA256(source_tree ∥ config ∥ dependencies ∥ datasets ∥ seeds)
```

Example: `lgae-q-8f1c9d...`

Every artifact generated from that run must contain the exact same QID.

This makes it impossible to mix artifacts from different runs without
detection.

### 2.3 Qualification provenance DAG

```
source tree
    │
    ├──── config
    ├──── environment
    ├──── dataset
    └──── seed manifest
             │
             ▼
          execution
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
 tests    policy    geometry
     │       │        │
     └───────┼────────┘
             ▼
      qualification root
```

Hash each node. Commit the final root into the release manifest.

---

## Milestone 3 — Checkpoint Integrity (v5.3.3)

Current behavior: write hash, write Merkle root, load without
verification. This has little security value.

### 3.1 Verified loading pipeline

```
OPEN CHECKPOINT
      │
      ▼
parse manifest only
      │
      ▼
schema validation
      │
      ▼
expected file-set validation
      │
      ▼
hash every payload
      │
      ▼
compare expected hashes
      │
      ▼
recompute Merkle root
      │
      ▼
verify root
      │
      ▼
verify signature
      │
      ▼
deserialize
```

Nothing authoritative is parsed before integrity checks finish.

### 3.2 Explicit checkpoint schema

```json
{
  "checkpoint_schema": "LGAE_SAFE_CHECKPOINT_V2",
  "engine_version": "5.4.0",
  "created_at": "...",
  "files": {
    "tensors.safetensors": {
      "sha256": "...",
      "bytes": 123456
    },
    "graph.json": {
      "sha256": "...",
      "bytes": 2456
    }
  },
  "merkle_root": "...",
  "state_root": "...",
  "previous_checkpoint_root": "...",
  "authority_key_id": "...",
  "signature": "..."
}
```

### 3.3 Adversarial checkpoint tests

Tamper with: tensor data, graph state, controller state, governance
state, manifest hash, root hash, file size, file name, deleted file,
extra file, reordered entries, truncated file, altered signature,
wrong key.

Every one must fail closed. No warnings. No partial loading.

### 3.4 Remove unsafe pickle from trusted state

Replace `torch.save(...)` / `torch.load(..., weights_only=False)` for
structural-controller state where possible.

Store:
- model tensors → safetensors
- optimizer metadata → canonical JSON where possible
- configuration → canonical JSON
- RNG state → safe binary/JSON representation

If legacy pickle support remains:
```python
load_legacy_checkpoint(path, trusted_source=True)
```
must be explicit. Do not silently accept arbitrary pickle.

---

## Milestone 4 — Production Config Unification (v5.4.0)

### 4.1 Typed configuration models

Create `src/lgae_v3/config_schema.py` and `src/lgae_v3/config_profiles.py`
with immutable typed configuration models.

```python
class GovernanceThresholds:
    max_integral_lly_deficit: float
    max_operator_discrepancy: float
    max_cde_residual: float
    max_ph_drift: float
    ...
```

`ProductionConfig()` becomes the source of truth. Generate
`configs/v5_4_production.yaml` from the Python model.

Test:
```python
assert parse_yaml(generated_production_yaml) == ProductionConfig()
```

### 4.2 Production must fail closed

Production configuration must reject:
- `threshold = None`
- required diagnostic unavailable
- persistent homology unavailable
- crypto signer unavailable
- unsupported backend
- nondeterministic mode enabled
- unsafe checkpoint loader enabled

Unless explicitly operating under a documented degraded mode.

Do not silently convert safety enforcement into monitoring.

---

## Milestone 5 — Mutation Authority Enforcement (v5.4.0)

### 5.1 Authority levels with operational semantics

| Level | Examples | Evidence threshold |
|-------|----------|-------------------|
| `REVERSIBLE` | reweight, temporary gauge | Low |
| `STRUCTURAL` | add/prune edge, spawn temporary fiber | Medium |
| `IRREVERSIBLE` | collapse fiber, commit causal relation, delete bridge | High |

### 5.2 Evidence policy matrix

| Check | Reversible | Structural | Irreversible |
|-------|-----------|-----------|-------------|
| local curvature | Yes | Yes | Yes |
| multi-curvature | Optional | Yes | Yes |
| spectral delta | Optional | Yes | Yes |
| β0 connectivity | Yes | Yes | Yes |
| PH drift | No | Yes | Yes |
| multi-step rollout | No | Yes | Yes |
| uncertainty bound | Yes | Yes | Yes |
| counterfactual support | Optional | Yes | Yes |
| causal consistency | If relevant | Yes | Yes |
| global coverage | No | Conditional | Yes |
| signer required | No | Production | Yes |
| human/external authority | No | Optional | Policy-driven |

This must be executable configuration, not documentation.

### 5.3 Evidence object

Every proposal produces:

```python
MutationEvidence(
    action=...,
    authority=...,
    curvature=...,
    spectrum=...,
    topology=...,
    persistent_homology=...,
    uncertainty=...,
    rollout=...,
    causal=...,
    counterfactuals=...,
)
```

The governor evaluates the evidence object. It does not rerun arbitrary
hidden logic scattered across modules.

---

## Milestone 6 — Evidence/Certification Redesign (v5.4.0)

### 6.1 Semantic certification levels

Replace vague labels with evidence-derived levels:

| Level | Meaning |
|-------|---------|
| `UNCERTIFIED` | No evidence collected |
| `LOCAL_EVIDENCE` | Local checks only |
| `STRUCTURAL_EVIDENCE` | Local + structural checks |
| `GLOBAL_EVIDENCE` | All checks completed globally |
| `FORMALLY_REPLAYABLE` | Global + deterministic replay verified |

### 6.2 Evidence bitmap

```json
{
  "all_nodes_covered": true,
  "all_edges_covered": true,
  "exact_lly": true,
  "bakry_complete": true,
  "persistent_homology": true,
  "spectral_solver_converged": true,
  "fallback_used": false,
  "unavailable_checks": []
}
```

Certification level is calculated from actual completed evidence, not
from configuration counts.

---

## Milestone 7 — Equivariant State Encoder (v5.5.0)

Replace the purely global engineered feature vector with an equivariant
encoder.

### Node features

degree, weighted degree, local Ricci statistics, Bakry statistics,
CDE residual, stalk/fiber norm, node role, causal role, spectral
embedding, local uncertainty, centrality

### Edge features

affinity, length, Ollivier curvature, LLY curvature, Forman curvature,
transport/gauge descriptors, edge age, edge authority status, causal
metadata, bridge importance

### Global features

λ₂, spectral entropy, β₀, β₁, PH summary, curvature histograms, graph
density, topology signature, governance budget, system uncertainty

Pass through a permutation-equivariant graph encoder.

---

## Milestone 8 — Unified Q(S,a,target) (v5.5.0)

### Decomposed heads

```
Q = Q_U + ν·Q_IG - λ·Q_C - μ·Q_R - ρ·Q_D
```

Where:
- `Q_U`: expected downstream utility
- `Q_IG`: information gain
- `Q_C`: computational/structural cost
- `Q_R`: risk
- `Q_D`: structural disruption

**Note on coefficients:** ν, λ, μ, ρ are hyperparameters that
reintroduce a tuning surface. They should either be learned from
reward decomposition (not tuned against benchmark metrics) or fixed at
principled defaults and reported honestly. If tuned against the
benchmark, we're back to reverse-engineering the test.

Keep components separately observable rather than training one opaque
scalar.

### Uncertainty first-class

Implement at least one of: ensemble Q networks, bootstrap heads,
evidential prediction, MC dropout, distributional Q.

Prefer lightweight bootstrap heads initially.

Return `μ_Q, σ_Q`. Risk-adjust: `Q_safe = μ_Q - κ·σ_Q` for destructive
mutations.

---

## Milestone 9 — Hierarchical Candidate Generator (v5.5.0)

Do not score every possible edge on a large graph (O(N²)).

Two-stage:

```
Stage 1: ANN / topology heuristic / uncertainty / curvature
         → top-K candidate regions (K=512)

Stage 2: learned Q evaluator
         → top-M structural actions (M=32)

Stage 3: counterfactual MPC
         → top-M sequences (M=8)

Stage 4: governor certification
```

This is the scalable path.

---

## Milestone 10 — MPC Integration (v5.5.0)

Integrate `StructuralMPC` into the live decision path.

For each high-value candidate:

```
state S0
  │
  ├── action a1 → S1
  │      │
  │      ├── a2 → S2
  │      └── ...
  │
  └── alternative sequences
```

Evaluate:
```
J = Σ_{t=0}^{H} γ^t (U_t + ν·IG_t - λ·C_t - μ·R_t)
```

Short horizons first: H = 2-4. Use beam search, learned value tail,
pruning, batched simulation.

---

## Milestone 11 — Procedural Environment Generator (v5.6.0)

### Graph families

Erdős-Rényi, Barabási-Albert, Watts-Strogatz, SBM, hierarchical SBM,
trees, grids, expanders, small-world, scale-free, geometric graphs,
hypergraphs, causal DAGs, temporal graphs, heterophilic graphs,
agent networks, knowledge graphs

### Varied parameters

node count, edge density, community structure, noise, curvature,
bridge structure, metric distortion, fiber dimension, gauge condition,
fault patterns

### Procedural interventions

Controlled defects: bridge failure, oversquashing bottleneck,
redundant edge, bad affinity, bad metric length, gauge corruption,
fiber mismatch, causal contradiction, community disconnection,
spectral collapse, topological hole, high-curvature bottleneck

Because the defect is injected, the ground truth intervention is known.
This gives clean counterfactual supervision.

### Stop relying on semantic labels

Instead of `correct_actions = {ADD_EDGE, NO_OP}`, compute all feasible
interventions and define:

```
a* = argmax_a ΔU(S,a)
```

or multi-objective:

```
a* = argmax_a J(S,a)
```

Semantic labels remain useful for diagnostics but are not the ultimate
oracle. This fixes the Task G ambiguity discovered in v5.3.2.

---

## Milestone 12 — Offline Large-Scale Training (v5.6.0)

### Counterfactual replay buffer

Each transition:

```json
{
  "state_id": "...",
  "state_features": {},
  "action": {},
  "authority": "...",
  "next_state_id": "...",
  "delta_utility": 0.0,
  "information_gain": 0.0,
  "risk": 0.0,
  "cost": 0.0,
  "governor_result": "...",
  "uncertainty": 0.0
}
```

Store rejected actions too — they provide valuable negative examples.

### Offline training pipeline

```
procedural simulation
      ↓
counterfactual dataset
      ↓
offline Q training
      ↓
held-out evaluation
      ↓
shadow deployment
      ↓
governed online learning
```

Do not let an online RL controller modify important graphs before
offline training and shadow validation.

---

## Milestone 13 — Generalization Qualification (v5.6.0)

### Split by graph family

```
TRAIN:       ER, BA, grid, SBM
VALIDATION:  different sizes and parameters
HELD-OUT:    tree, hierarchical SBM, Watts-Strogatz
OOD:         real graphs
```

### Report

- ID accuracy
- ID regret
- structural holdout regret
- topology-family holdout regret
- size extrapolation regret
- noise robustness

The learned system should not pass merely because it memorizes graph
archetypes.

---

## Milestone 14 — Real-World Structural Benchmarks (v5.6.0)

Karate Club remains only a smoke test.

### Oversquashing

Modify graphs before GNN message passing. Measure: accuracy,
long-range information propagation, effective resistance, spectral
properties, curvature, number of modifications.

Compare to: no rewiring, random, SDRF, FoSR-style methods, spectral
heuristics.

### Agent communication topology

Simulated agents with information propagation under: latency
constraints, bandwidth limits, node failures, partitioning, dynamic
tasks. Let LGAE modify the communication graph. Measure: task
completion, message count, latency, fault tolerance, communication
cost.

### Memory graph restructuring

Synthetic or real knowledge/memory graph. Test whether LGAE can: add
useful retrieval edges, prune redundant relations, preserve semantic
communities, reduce retrieval hops, avoid catastrophic topology changes.

### Network resilience

Inject: node failure, edge failure, bandwidth degradation, partition
threats. Allow LGAE to restructure.

**Note:** The real-world performance gate (success on ≥3 unrelated
domains with substantial margin) is a research outcome, not an
engineering milestone. It cannot be scheduled. If the controller does
not generalize, the engineering milestones still shipped value.

---

## Milestone 15 — Transactional Durable Ledger (v5.6.0)

### Durable governance ledger

```json
{
  "schema": "LGAE_MUTATION_RECEIPT_V2",
  "sequence": 184291,
  "parent_receipt_hash": "...",
  "state_before": "...",
  "proposal_hash": "...",
  "action": {},
  "authority": "STRUCTURAL",
  "evidence_root": "...",
  "decision": "ACCEPT",
  "state_after": "...",
  "controller_version": "...",
  "governor_version": "...",
  "policy_hash": "...",
  "timestamp": "...",
  "node_identity": "...",
  "signature": "..."
}
```

Hash chain: `R_n = H(R_{n-1} ∥ payload_n)`. Ed25519-sign `R_n`.

### Crash-safe transactional commits

```
proposal
   ↓
shadow state
   ↓
evaluation
   ↓
governor decision
   ↓
prepare receipt
   ↓
write-ahead log
   ↓
fsync
   ↓
commit state
   ↓
fsync
   ↓
finalize receipt
```

On startup: replay WAL, verify receipt chain, verify state root,
recover incomplete transaction.

### Concurrency (deferred)

Optimistic concurrency with `expected_state_root` is the right design
for multi-worker deployment. However, there is no multi-worker
deployment today. Single-process governance with WAL and crash recovery
is sufficient for v5.4.0. Add concurrency when there's a concrete
multi-worker requirement.

---

## Milestone 16 — Shadow Deployment (v5.6.0+)

Before autonomous structural mutation:

```
controller proposes
governor evaluates
result logged
BUT
authoritative state unchanged
```

Compare choices to: heuristic, current production policy, oracle where
available.

Run shadow mode until it establishes meaningful advantage.

---

## Milestone 17 — Governed Online Adaptation (v5.6.0+)

### Controlled activation ladder

| Level | Behavior |
|-------|----------|
| 0 | Observation only |
| 1 | Propose-only |
| 2 | Automatically perform reversible changes |
| 3 | Governed structural modifications |
| 4 | High-impact with strong certification |
| 5 | Online self-improvement with rollback and authority controls |

Do not jump directly to Level 5.

### Rollback

Every committed mutation has either:
- an explicit inverse
- a previous checkpoint
- an event replay path

### Online learning isolation

```
active model A
      │
experience
      ▼
candidate model B
      │
shadow qualification
      ▼
promotion gate
      │
PASS ──→ B becomes active
FAIL ──→ discard B
```

Every promoted model gets: model hash, training-data root,
qualification root, parent model, promotion receipt.

### Model regression gates

A candidate controller cannot be promoted unless:

```
R_new ≤ R_old
```

within confidence bounds across: ID, held-out, OOD, adversarial,
safety, real-world.

Require no safety regression even if average reward improves.

---

## Milestone 18 — Production Qualification (v5.6.0+)

### Release structure

```
LGAE-v5.4.0/
│
├── src/
├── tests/
├── configs/
├── scripts/
├── qualification/
│   ├── manifest.json
│   ├── tests.json
│   ├── geometry.json
│   ├── policy.json
│   ├── generalization.json
│   ├── production.json
│   └── benchmark.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GOVERNANCE.md
│   ├── SECURITY.md
│   ├── REPRODUCIBILITY.md
│   ├── CHECKPOINT_FORMAT.md
│   └── QUALIFICATION.md
├── artifacts/
│   └── lgae-5.4.0-py3-none-any.whl
├── MANIFEST.sha256
├── RELEASE_ROOT
└── RELEASE_SIGNATURE
```

### Release gates

**Integrity:**
- 0 inconsistent qualification metrics
- 0 unverified checkpoint paths
- 0 nondeterministic semantic ordering
- 0 Python hash dependencies

**Tests:**
- 100% critical test pass
- multiple PYTHONHASHSEED pass
- repeated deterministic qualification pass
- tamper suite pass
- restart/recovery suite pass

**Policy:**
- Learned controller beats random, cheap heuristic, and current
  StructuralExecutive on held-out regret with substantial margin

**Real-world:**
- Success on at least 3 unrelated domains (research gate — may fail)

**Security:**
- Tampered checkpoint rejected
- Tampered ledger rejected
- Wrong signer rejected
- Replayed receipt rejected
- Stale state mutation rejected
- Unsafe pickle unavailable by default
- Malformed state rejected

**Operational:**
- Crash during mutation → recovery
- Crash during checkpoint → recovery
- Disk-full → graceful failure
- Process kill → recovery
- Corrupted latest checkpoint → fallback
- GPU failure → CPU fallback

---

## Testing Hierarchy

### Unit tests

Every mathematical primitive.

### Property-based tests (incremental)

Start with metamorphic tests (plain pytest), then add Hypothesis for
mathematical primitives where it adds the most value:
- SO(d) orthogonality
- det(U) ≈ +1
- edge lengths remain positive
- state hashes stable under serialization
- node permutation preserves equivariant output
- rejected transaction leaves no mutation

### Metamorphic tests

- Permuting node IDs should not change structural decisions except for
  corresponding target permutations
- Scaling all affinities consistently should preserve relevant
  normalized quantities
- Serialization round-trip: `S = deserialize(serialize(S))`
- Checkpoint restore yields identical `state_root`

### Adversarial tests

Inject: NaN, Inf, huge weights, negative lengths, duplicate edges,
stale slots, generation mismatch, corrupted gauge, ill-conditioned
transport, singular Laplacian, disconnected graphs, empty graphs,
one-node graphs, maximum-capacity graphs.

System fails deterministically.

### Governance invariant tests

- Rejected proposal cannot alter state
- Quarantined proposal cannot alter authoritative state
- Every accepted mutation has exactly one valid receipt
- Receipt's before-root equals previous state
- Receipt's after-root equals current state
- State cannot advance without authority
- Irreversible action cannot use reversible evidence policy

These should be among the strongest tests in the repository.

---

## Capability Maturity Levels

| Level | Modules |
|-------|---------|
| **CORE** | GraphBuffers, GeometryGovernor, safe checkpointing, receipt ledger, state hashing, geometry operators |
| **INTEGRATED** | Equivariant encoder, Q controller, MPC, candidate generator, uncertainty |
| **EXPERIMENTAL** | Hypergraphs, causal structures, advanced contextual gauge models, exotic topology operators |

Experimental existence must not be confused with supported capability.

---

## Performance (after correctness)

### Do not optimize torch.compile yet

First establish a profiler. Measure: candidate generation, graph
encoder, Q evaluation, counterfactual simulation, curvature, persistent
homology, spectral solver, governor, serialization, receipt generation.

Then optimize the top two bottlenecks.

### Architecture

```
fast path:  approximate local diagnostics
slow path:  exact/global certification
```

A reversible action may use the fast path. An irreversible action may
trigger expensive global analysis.

### Caching (higher priority than torch.compile)

Track dirty regions. For a mutation involving (u,v), invalidate:
u, v, their neighborhoods, affected curvature edges, affected local
operators. Reuse the rest.

This can create much larger gains than torch.compile.

### Incremental spectral computation

Warm-start LOBPCG, reuse previous eigenvectors, low-rank update
approximations. Trigger full recomputation periodically or for
high-authority mutations.

### Incremental topology

Tiered checks:

```
cheap β0 / bridge test
        ↓
local topology signatures
        ↓
approximate PH
        ↓
full PH for high-authority mutation
```

Evidence strength scales with mutation risk.

---

## Observability

Structured runtime metrics:

proposals/sec, accept %, reject %, quarantine %, mean evidence latency,
policy uncertainty, Q regret proxy, curvature distribution, spectral
gap, PH drift, state version, checkpoint age, receipt-chain health,
candidate recall, MPC depth.

Prometheus/OpenTelemetry-compatible output eventually.

---

## What Not to Build Yet

Do not prioritize:

- more curvature variants
- more exotic manifolds
- larger neural networks
- huge LLM integration
- distributed multi-node runtime
- more hypergraph complexity
- extra visualization
- aggressive torch.compile
- dozens of new action types

The bottleneck is decision quality, not mathematical sensor count.

---

## Implementation Sequence

```
MILESTONE 1  Reproducibility repair              ← v5.3.3
MILESTONE 2  Release evidence rebuild             ← v5.3.3
MILESTONE 3  Checkpoint/Merkle verification       ← v5.3.3
MILESTONE 4  Production config unification        ← v5.4.0
MILESTONE 5  Mutation authority enforcement       ← v5.4.0
MILESTONE 6  Evidence/certification redesign      ← v5.4.0
MILESTONE 7  Equivariant state encoder            ← v5.5.0
MILESTONE 8  Unified Q(S,a,target)                ← v5.5.0
MILESTONE 9  Hierarchical candidate generator     ← v5.5.0
MILESTONE 10 MPC integration                      ← v5.5.0
MILESTONE 11 Procedural counterfactual env        ← v5.6.0+
MILESTONE 12 Offline large-scale training         ← v5.6.0+
MILESTONE 13 Generalization qualification         ← v5.6.0+
MILESTONE 14 Real-world structural benchmarks     ← v5.6.0+
MILESTONE 15 Transactional durable ledger         ← v5.6.0+
MILESTONE 16 Shadow deployment                    ← v5.6.0+
MILESTONE 17 Governed online adaptation           ← v5.6.0+
MILESTONE 18 Production qualification             ← v5.6.0+
```

Do not reorder substantially.

---

## The Architectural Change That Matters

The strongest future form of LGAE is not:

> a neural network that edits graphs.

It is:

> a structural decision system where learned models search for useful
> graph interventions, geometric operators quantify their consequences,
> and a separate cryptographically auditable authority layer determines
> whether those interventions may affect persistent state.

```
learned intelligence
        +
geometric world model
        +
counterfactual planning
        +
uncertainty
        +
formal governance
        +
cryptographic provenance
        +
reversible state evolution
```

The geometric machinery already gives LGAE an unusually strong substrate.
The next build should concentrate almost entirely on making the
structural learner worthy of that substrate and making the governance
claims mechanically true.

---

*This roadmap is a living document. It will be updated as milestones
are completed and as research findings inform the plan.*
