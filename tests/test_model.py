import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.model import TSPModel, TSPEncoder, TSPDecoder


@pytest.fixture
def model():
    return TSPModel(d_model=64, n_heads=4, n_encoder_layers=2, d_ff=128)


def test_encoder_output_shapes():
    enc = TSPEncoder(d_model=64, n_heads=4, n_layers=2, d_ff=128)
    cities = torch.rand(8, 20, 2)
    node_emb, graph_emb = enc(cities)
    assert node_emb.shape == (8, 20, 64)
    assert graph_emb.shape == (8, 64)


def test_model_tour_shape(model):
    cities = torch.rand(16, 20, 2)
    tour, log_probs = model(cities)
    assert tour.shape == (16, 20)
    assert log_probs.shape == (16,)


def test_tour_is_valid_permutation(model):
    B, N = 8, 20
    cities = torch.rand(B, N, 2)
    tour, _ = model(cities)
    for i in range(B):
        assert sorted(tour[i].tolist()) == list(range(N)), f"invalid tour at {i}"


def test_greedy_tour_valid(model):
    B, N = 4, 15
    cities = torch.rand(B, N, 2)
    tour, _ = model.greedy_tour(cities)
    for i in range(B):
        assert sorted(tour[i].tolist()) == list(range(N))


def test_no_nans(model):
    cities = torch.rand(4, 20, 2)
    tour, log_probs = model(cities)
    assert not torch.isnan(log_probs).any(), "NaN in log_probs"
    assert not torch.isinf(log_probs).any(), "Inf in log_probs"


def test_masking_no_repeat(model):
    """Stochastic rollout must never revisit a city."""
    B, N = 32, 20
    cities = torch.rand(B, N, 2)
    for _ in range(5):  # multiple stochastic runs
        tour, _ = model(cities)
        for i in range(B):
            assert len(set(tour[i].tolist())) == N


def test_log_probs_negative(model):
    cities = torch.rand(4, 20, 2)
    _, log_probs = model(cities)
    assert (log_probs <= 0).all(), "log-probs should be ≤ 0"


def test_different_n_cities(model):
    for n in [10, 15, 30]:
        cities = torch.rand(4, n, 2)
        tour, lp = model(cities)
        assert tour.shape == (4, n)
        for i in range(4):
            assert sorted(tour[i].tolist()) == list(range(n))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
