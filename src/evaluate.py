"""Optimality-gap evaluation against heuristic baselines."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from typing import Dict, Tuple
from src.env import tour_length, sample_instances
from src.heuristics import batch_nn_tours, batch_best_tours, random_tours
from src.model import TSPModel


def optimality_gap(model_len: np.ndarray, baseline_len: np.ndarray) -> Tuple[float, float]:
    """Mean ± std of (model/baseline - 1) * 100%."""
    gap = (model_len / baseline_len - 1.0) * 100.0
    return float(gap.mean()), float(gap.std())


@torch.no_grad()
def evaluate_model(
    model: TSPModel,
    cities: torch.Tensor,
    batch_size: int = 256,
) -> Dict[str, object]:
    """
    Evaluate model on cities in mini-batches.
    Returns dict with tour lengths and gaps vs NN, best-heuristic, random.
    """
    model.eval()
    device = next(model.parameters()).device
    N_total = cities.shape[0]

    model_lens, nn_lens, best_lens, rand_lens = [], [], [], []
    method = "2-opt"

    for start in range(0, N_total, batch_size):
        batch = cities[start:start + batch_size].to(device)
        tour, _ = model.greedy_tour(batch)
        model_lens.append(tour_length(batch, tour).cpu())

        nn_t = batch_nn_tours(batch)
        nn_lens.append(tour_length(batch, nn_t).cpu())

        best_t, method = batch_best_tours(batch)
        best_lens.append(tour_length(batch, best_t).cpu())

        rand_t = random_tours(batch)
        rand_lens.append(tour_length(batch, rand_t).cpu())

    model_len = torch.cat(model_lens).numpy()
    nn_len    = torch.cat(nn_lens).numpy()
    best_len  = torch.cat(best_lens).numpy()
    rand_len  = torch.cat(rand_lens).numpy()

    gap_nn_mean,   gap_nn_std   = optimality_gap(model_len, nn_len)
    gap_best_mean, gap_best_std = optimality_gap(model_len, best_len)
    gap_rand_mean, gap_rand_std = optimality_gap(model_len, rand_len)

    return {
        "model_len_mean": float(model_len.mean()),
        "model_len_std":  float(model_len.std()),
        "gap_vs_nn_mean": gap_nn_mean,
        "gap_vs_nn_std":  gap_nn_std,
        f"gap_vs_{method}_mean": gap_best_mean,
        f"gap_vs_{method}_std":  gap_best_std,
        "gap_vs_random_mean": gap_rand_mean,
        "gap_vs_random_std":  gap_rand_std,
        "baseline_method": method,
        "n_instances": N_total,
    }


if __name__ == "__main__":
    from src.env import set_seed

    set_seed(42)
    # tiny sanity check: on 3 hand-crafted instances verify gap math
    # cities on a square: NN tour = 4.0, model (random init) will be worse
    cities = torch.tensor([
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
    ])  # (1,4,2)
    nn_t = batch_nn_tours(cities)
    nn_l = tour_length(cities, nn_t).numpy()
    model_l = np.array([4.5])  # pretend model did worse
    g_mean, g_std = optimality_gap(model_l, nn_l)
    expected_gap = (4.5 / 4.0 - 1) * 100  # 12.5%
    print(f"gap={g_mean:.2f}%  expected={expected_gap:.2f}%  ok={abs(g_mean-expected_gap)<0.01}")
