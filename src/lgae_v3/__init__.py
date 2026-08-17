"""LGAE-v3: geometry-governed self-evolving graph/latent controller."""
from .config import LGAEConfig, load_config, config_structural_hash, config_governance_hash, ProductionConfig, ResearchConfig
from .evolution import LGAEEngine
from .fibers import FixedWidthFiberLatent, FiberController, SOConnectionBank, project_to_so_d
from .operators import DualOperatorState, SparseDualOperatorState
from .types import (
    EdgeRole,
    GraphBuffers,
    make_graph_buffers,
    make_bucketed_graph_buffers,
    round_edge_capacity,
    MutationDecision,
    MutationResult,
    CertificationLevel,
)
from .training import (
    LGAETrainCore, train_step, padded_markov_edges, refresh_padded_markov_edges_,
    padded_markov_edges_with_slots, refresh_padded_markov_edges_with_slots_,
)
from .governor import GeometryGovernor
from .mutations import (
    AddEdge, ReweightEdge, ReweightAffinity, ReweightLength, CoupledReweight,
    PruneEdge, RicciFlowReweight, MutationCooldownTracker,
    StructuralMutation, GraphMutation,
    MutationAuthorityLevel, mutation_authority_level,
    mutation_to_spec, mutation_from_spec,
)
from .neighbor_index import (
    NeighborIndex, ExactChunkedKNN, KNNGraphResult,
    build_knn_graph, recall_at_k,
)
from .executive import (
    StructuralExecutive, ExecutiveNetwork, ActionProposal, StructuralObservation,
    StructuralAction, ACTION_LIST, ACTION_TO_IDX, NUM_ACTIONS,
)
from .uncertainty import (
    EnsembleUncertainty, ConformalCalibrator, UncertaintyEstimate,
    uncertainty_gated_decision,
)
from .credit import (
    MutationCreditTracker, MutationReceipt, MutationOutcome,
)
from .consolidation import (
    StabilityPlasticityController, FiberState, FiberLifecycleStage, CapacityBudget,
)
from .counterfactual import (
    StructuralCounterfactualEngine, CounterfactualResult,
)
from .structural_loop import (
    StructuralLearningLoop, StructuralLoopResult,
)
from .action_bridge import (
    action_to_mutation, certify_action_through_governor, ActionBridgeResult,
)
from .dynamic_gauge import (
    DynamicGaugeNetwork, DynamicGaugeBank, StaticGaugeAdapter,
    gauge_transport, gauge_alignment_loss,
)
from .timescales import (
    Timescale, TimescaleSchedule, AdaptationState, MultiTimescaleController,
)
from .sheaf_diffusion import (
    sheaf_laplacian_diffusion, sheaf_adjacency_diffusion,
    gated_sheaf_diffusion, agreement_gate, compare_diffusion_methods,
    gauge_orthogonality_penalty,
)
from .ann_index import (
    ANNNeighborIndex, FAISSIndex, RandomProjectionANN, HNSWIndexNumpy,
)
from .production_dynamics import (
    CurvatureHysteresisController, LatentEquilibriumBarrier, GraphHashBaseline, GraphFeatureBaseline,
)
from .transactions import GraphTransaction, graph_transaction
from .receipts import (
    mutation_receipt, append_receipt, verify_receipt_chain,
    ed25519_available, generate_keypair, sign_receipt, verify_receipt_signature,
)
from .causal_edges import (
    EdgeSemantics, CausalEdge, CausalEdgeRegistry, infer_causality_from_temporal,
)
from .topology import (
    graphbuffers_to_networkx, topology_signature, topology_drift,
    topology_signature_buffers, find_bridges_buffers,
)
from .mpc import StructuralMPC, MPCPlanResult
from .equivariant import (
    EquivariantExecutiveNetwork, MessagePassingLayer,
    graphbuffers_to_edge_index, permutation_invariance_test,
)
from .hypergraph import (
    Hyperedge, HypergraphBuffers, hypergraph_laplacian_diffusion,
    clique_expansion, star_expansion,
)
from .deterministic import (
    DeterministicRNGContext, deterministic_mode, derive_seed,
)
from .reproducibility import ReproducibilityInfo, qualification_id

__all__ = [
    "LGAEConfig", "load_config", "config_structural_hash", "config_governance_hash", "ProductionConfig", "ResearchConfig",
    "LGAEEngine", "FixedWidthFiberLatent", "FiberController", "SOConnectionBank", "project_to_so_d",
    "DualOperatorState", "SparseDualOperatorState", "EdgeRole", "GraphBuffers", "make_graph_buffers", "make_bucketed_graph_buffers", "round_edge_capacity",
    "MutationDecision", "MutationResult", "CertificationLevel",
    "AddEdge", "ReweightEdge", "ReweightAffinity", "ReweightLength", "CoupledReweight",
    "PruneEdge", "RicciFlowReweight", "MutationCooldownTracker",
    "MutationAuthorityLevel", "mutation_authority_level",
    "mutation_to_spec", "mutation_from_spec",
    "LGAETrainCore", "train_step", "padded_markov_edges", "refresh_padded_markov_edges_", "padded_markov_edges_with_slots", "refresh_padded_markov_edges_with_slots_",
    "gauge_orthogonality_penalty", "ANNNeighborIndex", "FAISSIndex", "RandomProjectionANN",
    "CurvatureHysteresisController", "LatentEquilibriumBarrier", "GraphHashBaseline", "GraphFeatureBaseline",
    "GraphTransaction", "graph_transaction",
]
from .version import VERSION as __version__
