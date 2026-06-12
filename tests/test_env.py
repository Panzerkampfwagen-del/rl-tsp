import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest
from src.env import tour_length, sample_instances, DISTRIBUTIONS


def test_tour_length_3city():
    cities = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    tour = torch.tensor([[0, 1, 2]])
    length = tour_length(cities, tour).item()
    expected = 2.0 + 2.0 ** 0.5
    assert abs(length - expected) < 1e-5, f"got {length}, expected {expected}"


def test_tour_length_trivial():
    # Batch of 2, 4 cities on unit square corners
    cities = torch.tensor([
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
    ])
    tour = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]])
    lengths = tour_length(cities, tour)
    # Perimeter of unit square = 4.0 regardless of CW/CCW
    assert abs(lengths[0].item() - 4.0) < 1e-5
    assert abs(lengths[1].item() - 4.0) < 1e-5


def test_tour_length_batch():
    B, N = 32, 20
    cities = torch.rand(B, N, 2)
    tour = torch.stack([torch.randperm(N) for _ in range(B)])
    lengths = tour_length(cities, tour)
    assert lengths.shape == (B,)
    assert (lengths > 0).all()


@pytest.mark.parametrize("dist", list(DISTRIBUTIONS.keys()))
def test_distribution_shape(dist):
    cities = sample_instances(dist, batch=16, n=20)
    assert cities.shape == (16, 20, 2), f"{dist}: shape {cities.shape}"
    assert cities.dtype == torch.float32


@pytest.mark.parametrize("dist", list(DISTRIBUTIONS.keys()))
def test_distribution_range(dist):
    cities = sample_instances(dist, batch=64, n=20)
    assert cities.min().item() >= 0.0 - 1e-6
    assert cities.max().item() <= 1.0 + 1e-6


def test_seed_reproducibility():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = sample_instances("uniform", 8, 20, rng=rng1)
    b = sample_instances("uniform", 8, 20, rng=rng2)
    assert torch.allclose(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
