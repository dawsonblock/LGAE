# Changelog

## v5.3.3 — Reproducibility repair (current)

Milestone 1 of the v5.4.0 roadmap. Eliminates nondeterminism that caused
qualification results to vary across PYTHONHASHSEED values.

### Added

- **Canonical action ordering** (`ACTION_ORDER`, `ACTION_TO_INDEX`,
  `canonical_action()`): deterministic selection from action sets,
  replacing all `next(iter(set))` calls that were nondeterministic under
  PYTHONHASHSEED variation.
- **`DeterministicRNGContext`** (`src/lgae_v3/deterministic.py`):
  domain-separated substreams (graph_generation, model_initialization,
  counterfactuals, qualification, etc.) via SHA-256 seed derivation.
  Prevents one new random call from changing every subsequent result.
- **`deterministic_mode()`** context manager: seeds all global RNGs and
  enables torch deterministic algorithms.
- **`ReproducibilityInfo`** (`src/lgae_v3/reproducibility.py`): collects
  seed, PYTHONHASHSEED, source commit, source tree hash, config hash,
  platform, torch version. All qualification reports now include this
  metadata.
- **`qualification_id()`**: deterministic QID shared across all reports
  from the same run. Makes it impossible to mix artifacts from different
  runs without detection.
- **23 new reproducibility tests** (`tests/test_v533_reproducibility.py`).
- **Forensic baseline** (`qualification/baseline_v5_3_2.json`).
- **Roadmap document** (`docs/ROADMAP_V5_4.md`): 18-milestone plan for
  governed structural intelligence runtime.
- **Complete guide** (`docs/COMPLETE_GUIDE.md`): comprehensive
  documentation of architecture, audit responses, and limitations.

### Fixed

- **Oracle in `run_benchmark()`**: was using `next(iter(correct_actions()))`
  (nondeterministic), now uses `OracleController` which picks highest-ΔU
  action with canonical tie-breaking.
- **Policy training label selection**: was using `list(set)[0]`
  (nondeterministic), now uses canonical action ordering with highest-ΔU
  selection.
- **Removed all Python `hash()` from deterministic logic** in
  `counterfactual.py`: replaced with SHA-256-based `_stable_hash_u64()`.

### Verification

- 652 tests pass (was 629, +23 new)
- All 652 tests pass under PYTHONHASHSEED=0,1,2,42,123456
- All 3 qualification reports are byte-for-byte identical across repeated runs
- Policy qualification: 100% accuracy, 0.0 regret (deterministic)
- Archived baseline as `archive/v5.3.2-received` tag

## v5.3.2 — Research improvements: Q-learning, hierarchical retrieval, gauge norm control

This release implements the audit's central recommendation: improve
structural decision intelligence, not geometric measurement. The learned
executive now has a Q(S,a) controller trained on counterfactual outcomes,
a wider candidate retrieval pipeline, and several safety/stability upgrades.

### Added

- **Q(S,a) counterfactual controller** (`src/lgae_v3/benchmark/counterfactual.py`):
  generates (state, action, ΔU) triples across 8 topology families
  (path, cycle, grid, star, BA, WS, random, bipartite) and trains a
  Q-network via MSE regression. Policy derived as argmax_a Q(S,a) instead
  of supervised action classification. Held-out evaluation on 3 unseen
  families (wheel, lollipop, caveman) achieves 86% accuracy, vs 30% for
  the old supervised policy on structurally held-out tasks.
- **Runner script** (`scripts/train_q_controller.py`): end-to-end
  dataset generation, training, and held-out evaluation.
- **22 new tests** (`tests/test_v531_research_improvements.py`): cover
  GraphFeatureBaseline, hierarchical retrieval, gauge norm clamping,
  equilibrium residual, counterfactual dataset, and Q-network training.

### Changed

- **GraphFeatureBaseline replaces GraphHashBaseline** as the default
  credit-assignment value estimator. Uses online ridge regression on
  16-dim structural features (N, E, λ₂, curvature moments, degree moments,
  Betti numbers, latent variance). Similar graph states now produce
  similar baseline predictions, reducing variance in advantage estimation.
  GraphHashBaseline retained as fallback.
- **Hierarchical candidate retrieval** replaces top-24 node cap. New
  defaults: top-64 nodes + 4 latent-distance KNN per node, 512 max pairs.
  Correct endpoints outside the top-64 are now reachable via KNN.
- **Dynamic gauge generator norm clamping** (default max=1.0). Clamps
  ||A||_F after antisymmetrization, preserving A_ji = -A_ij exactly.
  Prevents Cayley conditioning problems and bounds the Jacobian of the
  state-dependent gauge map. Spectral norm available as opt-in.
- **Equilibrium barrier upgraded** from pure state-delta to combined
  state-delta + dynamics residual. Now checks ||F(z_t) - z_t|| / ||z_t||
  in addition to ||z_t - z_{t-1}|| / ||z_{t-1}||, catching periodic
  orbits and metastable plateaus the old check missed.

## v5.3.1 — Integrity & baseline-comparison fixes

This is a correctness/integrity patch on v5.3.0. It does not change the
governor, geometry oracles, or numerical kernels. It fixes four issues
identified in a review of v5.3.0's release artifacts and adds the missing
baseline comparison.

### Fixed

- **Nondeterministic policy qualification (release-gate integrity bug).**
  `qualify_structural_policy` constructed the `StructuralExecutive` (and its
  network/target-scorer weight initialization) *before* calling
  `torch.manual_seed`. Three runs of `scripts/qualify_policy.py` on the same
  code produced diagnosis accuracy of 100%, 90%, and 86.7% — i.e. the
  release-gate number was a random draw and a bad draw could fail the 80%
  threshold. The seed is now set before network construction, making the
  qualification deterministic (83.3% / 0.0176 regret on the current code).
  `src/lgae_v3/benchmark/policy_qualification.py`

- **Stale `example_output.txt`.** It reported `"version": "3.2.0"` from a
  pre-v5 release. Regenerated from the current CLI (`5.3.0`).

- **Stale qualification reports.** `policy_qualification_report.json`
  (86.7% / 0.0274), `qualification_report.json`, and
  `production_qualification_report.json` were regenerated from the current
  deterministic code.

- **Circular benchmark utility in Task A.** `TaskA_Bottleneck.utility`
  added `+0.1 * inter_cluster_edge_count`, a term that directly encoded the
  correct action's structural signature (the correct action *is* "add an
  inter-cluster edge"). The utility is now the pure spectral gap λ₂ — a
  physical graph invariant. The correct action still maximizes λ₂ because
  of the physics of bottlenecks, not because the utility was written to
  reward it. `src/lgae_v3/benchmark/tasks.py`

### Added

- **Baseline controllers** (`src/lgae_v3/benchmark/baselines.py`):
  `RandomActionController` (lower bound), `SpectralHeuristicController`
  (non-learned threshold rules on cheap observables — not tuned per task),
  and `OracleController` (upper bound). These let the learned executive be
  compared on the same axis instead of in a vacuum.

- **Baseline comparison script** (`scripts/compare_baselines.py`): runs
  random / spectral-heuristic / learned / oracle on both the in-distribution
  tasks and truly held-out structurally-distinct tasks, reporting diagnosis
  accuracy and mean regret for each.

  Findings on the current code (seed 0, 500 gradient steps):
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
  The learned executive beats random and the spectral heuristic
  in-distribution, but **loses to the spectral heuristic on held-out
  diagnosis accuracy** (30% vs 60%), though with lower regret (0.62 vs
  2.40), suggesting it defaults to safe NO_OP on unseen structures rather
  than misdiagnosing. This is the first honest generalization signal the
  benchmark has produced.

- **Truly held-out task variants** (`src/lgae_v3/benchmark/tasks.py`):
  `HeldOutBottleneck` (variable cluster size / bridge position) and
  `HeldOutSpuriousEdge` (variable graph size). The original "held-out seeds
  101–105" produced *identical* graph structures to seed 42 for all six
  tasks (only latent noise differed); these parametric variants generate
  structurally different graphs so held-out evaluation measures something
  beyond seed noise.

### Changed

- **README tone.** Removed the hardcoded "559/559 passing" badge (the count
  is code-dependent and shouldn't be hardcoded into a badge URL), added a
  "validation: synthetic-only" badge, and added a "Validation boundaries"
  section that states explicitly what is and is not claimed. The stale
  "96.7%" v5.2 claim and "86.7%" v5.3 claim were corrected.

- **Naming note.** README now documents that the release is LGAE-v5.3.0,
  the repo is `1LR`, and the Python dist/module is `lgae-v3` / `lgae_v3`
  (kept for import stability). `docs/ARCHITECTURE.md` and
  `docs/READING_LIST.md` updated to stop implying the current version is
  3.2.

### Not fixed (deliberately)

- The package/module name `lgae_v3` is **not** renamed. Renaming would
  touch every import in 53 source + 31 test files for no functional gain
  and high regression risk. The naming note in the README documents the
  historical reason instead.

- The remaining benchmark tasks (B–F) still have utilities that are
  constructed around the correct action's effect. Fully decoupling them
  requires defining independent downstream objectives (reconstruction,
  diffusion mixing) per task, which is a larger redesign left for a future
  release. The README's Validation boundaries section states this.
