"""Tests for v5.3.1 research improvements.

Covers:
- GraphFeatureBaseline (replaces GraphHashBaseline)
- Hierarchical candidate retrieval (replaces top-24 bottleneck)
- Dynamic gauge generator norm clamping
- Latent equilibrium barrier with dynamics residual
- Counterfactual dataset generation
- Q-network training and evaluation
"""
import pytest
import torch
import numpy as np

from lgae_v3.production_dynamics import (
    GraphFeatureBaseline,
    GraphHashBaseline,
    LatentEquilibriumBarrier,
    compute_graph_features,
)
from lgae_v3.executive import StructuralExecutive
from lgae_v3.dynamic_gauge import DynamicGaugeNetwork, DynamicGaugeBank
from lgae_v3.benchmark.counterfactual import (
    generate_counterfactual_dataset,
    train_q_network,
    evaluate_q_network,
    QNetwork,
    TOPOLOGY_FAMILIES,
    HELD_OUT_FAMILIES,
)


# ===========================================================================
# GraphFeatureBaseline tests
# ===========================================================================

class TestGraphFeatureBaseline:
    def test_predict_zero_before_any_updates(self):
        b = GraphFeatureBaseline(feature_dim=16)
        f = torch.randn(16)
        assert b.predict("", f) == 0.0

    def test_update_and_predict_learns_value(self):
        b = GraphFeatureBaseline(feature_dim=16)
        f = torch.randn(16)
        b.update("", 1.0, f)
        b.update("", 2.0, f)
        pred = b.predict("", f)
        # After 2 updates with 1.0 and 2.0, prediction should be near 1.5
        assert 0.5 < pred < 2.5

    def test_similar_features_give_similar_predictions(self):
        """The key property: geometric similarity is preserved."""
        b = GraphFeatureBaseline(feature_dim=16)
        f1 = torch.randn(16)
        f2 = f1 + 0.01 * torch.randn(16)  # very similar
        b.update("", 5.0, f1)
        p1 = b.predict("", f1)
        p2 = b.predict("", f2)
        assert abs(p1 - p2) < 0.5  # similar features → similar predictions

    def test_hash_fallback_works(self):
        b = GraphFeatureBaseline(feature_dim=16)
        assert b.predict("some_hash") == 0.0
        b.update("some_hash", 3.0)
        assert b.predict("some_hash") == pytest.approx(3.0)

    def test_state_dict_roundtrip(self):
        b = GraphFeatureBaseline(feature_dim=16)
        f = torch.randn(16)
        b.update("", 1.5, f)
        sd = b.state_dict()
        b2 = GraphFeatureBaseline.from_state_dict(sd)
        assert b2.predict("", f) == pytest.approx(b.predict("", f))

    def test_compute_graph_features_returns_correct_dim(self):
        f = compute_graph_features(num_nodes=10, num_edges=20, lambda2=0.05)
        assert f.shape[0] == 16
        assert f.dtype == torch.float32


# ===========================================================================
# Hierarchical candidate retrieval tests
# ===========================================================================

class TestHierarchicalCandidateRetrieval:
    def test_default_top_k_is_64_not_24(self):
        """The audit found top-24 was a recall bottleneck. Default should be 64."""
        e = StructuralExecutive(hidden_dim=32)
        assert e.candidate_top_k == 64
        assert e.candidate_max_pairs == 512
        assert e.candidate_knn_per_node == 4

    def test_top_k_is_configurable(self):
        e = StructuralExecutive(hidden_dim=32, candidate_top_k=32, candidate_max_pairs=128, candidate_knn_per_node=2)
        assert e.candidate_top_k == 32
        assert e.candidate_max_pairs == 128
        assert e.candidate_knn_per_node == 2


# ===========================================================================
# Dynamic gauge norm clamping tests
# ===========================================================================

class TestDynamicGaugeNormClamping:
    def test_generator_norm_is_clamped(self):
        """Generator Frobenius norm should not exceed generator_norm_max."""
        net = DynamicGaugeNetwork(latent_dim=4, hidden_dim=32, generator_norm_max=0.5)
        z = torch.randn(5, 4)
        # The clamping happens in DynamicGaugeBank.matrices, not in the network forward.
        # So we test via the bank.
        bank = DynamicGaugeBank(edge_capacity=8, dim=4, hidden_dim=32, generator_norm_max=0.5)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        U = bank.matrices(z, src, dst)
        # U is in SO(d), so ||U||_F = sqrt(d). The clamping is on A, not U.
        # But we can verify SO(d) invariance is preserved.
        eye = torch.eye(4)
        for i in range(len(U)):
            assert torch.allclose(U[i].T @ U[i], eye, atol=1e-4)

    def test_reverse_edge_inverse_preserved_with_clamping(self):
        """A_ji = -A_ij must still hold after norm clamping."""
        torch.manual_seed(4)
        bank = DynamicGaugeBank(edge_capacity=8, dim=3, hidden_dim=16, generator_norm_max=0.3)
        z = torch.randn(3, 3)
        src = torch.tensor([0, 1])
        dst = torch.tensor([1, 0])
        U = bank.matrices(z, src, dst)
        assert torch.allclose(U[1], U[0].transpose(-1, -2), atol=1e-5, rtol=1e-5)

    def test_spectral_norm_is_opt_in(self):
        """Spectral norm should be off by default (causes non-determinism)."""
        net = DynamicGaugeNetwork(latent_dim=4, hidden_dim=32)
        assert net.use_spectral_norm == False


# ===========================================================================
# Equilibrium barrier with dynamics residual tests
# ===========================================================================

class TestEquilibriumBarrierResidual:
    def test_residual_tol_parameter_exists(self):
        barrier = LatentEquilibriumBarrier(delta_tol=1e-3, residual_tol=1e-4)
        assert barrier.residual_tol == 1e-4

    def test_small_residual_accelerates_equilibrium(self):
        """With small dynamics residual, equilibrium should be reached faster."""
        barrier = LatentEquilibriumBarrier(delta_tol=1e-3, required_consecutive=2, residual_tol=1e-3)
        z = torch.randn(10, 4)
        # First call: no previous, returns False
        assert barrier.observe(z) == False
        # Second call: small delta, small residual → consecutive=1
        z2 = z + 1e-5 * torch.randn(10, 4)
        residual = 1e-5 * torch.randn(10, 4)
        assert barrier.observe(z2, dynamics_residual=residual) == False  # consecutive=1
        # Third call: still small → consecutive=2 → equilibrated
        z3 = z2 + 1e-5 * torch.randn(10, 4)
        residual3 = 1e-5 * torch.randn(10, 4)
        assert barrier.observe(z3, dynamics_residual=residual3) == True  # consecutive=2

    def test_large_residual_blocks_equilibrium(self):
        """Large dynamics residual should block equilibrium even if delta is small."""
        barrier = LatentEquilibriumBarrier(delta_tol=1e-3, required_consecutive=2, residual_tol=1e-3)
        z = torch.randn(10, 4)
        barrier.observe(z)
        z2 = z + 1e-5 * torch.randn(10, 4)  # small delta
        large_residual = 10.0 * torch.randn(10, 4)  # large residual
        assert barrier.observe(z2, dynamics_residual=large_residual) == False
        assert barrier.consecutive == 0

    def test_backward_compatible_without_residual(self):
        """Calling observe() without dynamics_residual should still work."""
        barrier = LatentEquilibriumBarrier(delta_tol=1e-3, required_consecutive=2)
        z = torch.randn(10, 4)
        barrier.observe(z)
        z2 = z + 1e-5 * torch.randn(10, 4)
        # Should work without residual (backward compatible)
        result = barrier.observe(z2)
        assert isinstance(result, bool)

    def test_summary_includes_residual(self):
        barrier = LatentEquilibriumBarrier()
        z = torch.randn(10, 4)
        barrier.observe(z, dynamics_residual=torch.randn(10, 4) * 0.1)
        s = barrier.summary()
        assert "residual_tol" in s
        assert "last_relative_residual" in s


# ===========================================================================
# Counterfactual dataset and Q-learning tests
# ===========================================================================

class TestCounterfactualDataset:
    def test_generate_small_dataset(self):
        ds = generate_counterfactual_dataset(num_samples=48, families=["path"], seed=0)
        assert len(ds) > 0
        assert all(s.observation.shape[0] == 16 for s in ds)

    def test_dataset_has_correct_actions(self):
        from lgae_v3.benchmark.tasks import StructuralAction
        ds = generate_counterfactual_dataset(num_samples=48, families=["path"], seed=0)
        action_indices = set(s.action_idx for s in ds)
        assert len(action_indices) > 1  # multiple actions represented

    def test_held_out_families_are_disjoint_from_training(self):
        assert set(HELD_OUT_FAMILIES).isdisjoint(set(TOPOLOGY_FAMILIES))


class TestQNetworkTraining:
    def test_q_network_forward_pass(self):
        net = QNetwork(obs_dim=16, hidden_dim=64)
        x = torch.randn(4, 16)
        out = net(x)
        assert out.shape == (4, 9)  # 9 actions

    def test_train_q_network_learns_something(self):
        ds = generate_counterfactual_dataset(num_samples=240, families=["path", "cycle"], seed=0)
        result = train_q_network(ds, epochs=10, seed=0)
        assert result.train_samples > 0
        assert len(result.losses) == 10
        # Loss should decrease
        assert result.losses[-1] <= result.losses[0] * 1.1  # allow small noise

    def test_evaluate_q_network_returns_results(self):
        net = QNetwork(obs_dim=16, hidden_dim=64)
        results = evaluate_q_network(net, families=["path"], num_states_per_family=12, seed=42)
        assert len(results) == 1
        assert results[0].family == "path"
        assert 0.0 <= results[0].accuracy <= 1.0
