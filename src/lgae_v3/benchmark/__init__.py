"""v5.0 Task-grounded benchmark harness.

Synthetic tasks with known-optimal structural changes, designed to answer:
"Does LGAE's self-modification actually help?"

Metrics:
- Structural diagnosis accuracy: does the system identify the correct intervention?
- Mutation regret: R_t = U(m_t*) - U(m_t)
"""
from __future__ import annotations

from .tasks import (
    BenchmarkTask,
    StructuralAction,
    TaskState,
    TaskOutcome,
    TaskA_Bottleneck,
    TaskB_RepComplexity,
    TaskC_SpuriousEdge,
    TaskD_GaugeMismatch,
    TaskE_DistributionShift,
    TaskF_NoOp,
    ALL_TASKS,
    HeldOutBottleneck,
    HeldOutSpuriousEdge,
    heldout_tasks,
)
from .metrics import (
    StructuralDiagnosisResult,
    MutationRegretResult,
    BenchmarkResult,
    evaluate_diagnosis_accuracy,
    evaluate_mutation_regret,
    run_benchmark,
)
from .baselines import (
    RandomActionController,
    SpectralHeuristicController,
    OracleController,
    ALL_BASELINES,
)
from .harness import BenchmarkHarness

__all__ = [
    "BenchmarkTask", "StructuralAction", "TaskState", "TaskOutcome",
    "TaskA_Bottleneck", "TaskB_RepComplexity",
    "TaskC_SpuriousEdge", "TaskD_GaugeMismatch", "TaskE_DistributionShift",
    "TaskF_NoOp", "ALL_TASKS",
    "HeldOutBottleneck", "HeldOutSpuriousEdge", "heldout_tasks",
    "StructuralDiagnosisResult", "MutationRegretResult", "BenchmarkResult",
    "evaluate_diagnosis_accuracy", "evaluate_mutation_regret", "run_benchmark",
    "RandomActionController", "SpectralHeuristicController", "OracleController",
    "ALL_BASELINES",
    "BenchmarkHarness",
]
