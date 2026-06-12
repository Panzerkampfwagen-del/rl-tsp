"""Greedy rollout baseline with statistically-gated update (Kool et al.)."""

import copy
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Tuple
from src.env import tour_length, sample_uniform


class GreedyRolloutBaseline:
    """
    Maintains a frozen copy of the policy. Computes baseline lengths via greedy
    rollout. Updates the frozen copy only when the current policy is significantly
    better (paired t-test, p < threshold).
    """

    def __init__(self, model: nn.Module, n_cities: int, device: str,
                 val_size: int = 1000, p_threshold: float = 0.05) -> None:
        self.model = model
        self.baseline_model = copy.deepcopy(model).to(device)
        self.baseline_model.eval()
        self.n_cities = n_cities
        self.device = device
        self.val_size = val_size
        self.p_threshold = p_threshold

    @torch.no_grad()
    def get_baseline(self, cities: torch.Tensor) -> torch.Tensor:
        """Greedy rollout from the frozen baseline model. Returns lengths (B,)."""
        tour, _ = self.baseline_model.greedy_tour(cities)
        return tour_length(cities, tour)

    def update_if_better(self, val_seed: int = 999) -> Tuple[bool, float]:
        """
        Evaluate current model vs baseline on a fresh val set.
        Update baseline if paired t-test shows current is significantly better.
        Returns (updated, p_value).
        """
        rng = np.random.default_rng(val_seed)
        val_cities = sample_uniform(self.val_size, self.n_cities,
                                    device=self.device, rng=rng)

        self.baseline_model.eval()
        self.model.eval()

        with torch.no_grad():
            cur_tour, _ = self.model.greedy_tour(val_cities)
            bl_tour, _  = self.baseline_model.greedy_tour(val_cities)
            cur_len = tour_length(val_cities, cur_tour).cpu().numpy()
            bl_len  = tour_length(val_cities, bl_tour).cpu().numpy()

        # one-sided t-test: H1 = current is better (shorter)
        t_stat, p_value = stats.ttest_rel(cur_len, bl_len)
        updated = False
        if t_stat < 0 and p_value / 2 < self.p_threshold:
            self.baseline_model.load_state_dict(
                copy.deepcopy(self.model.state_dict())
            )
            updated = True

        self.model.train()
        return updated, float(p_value / 2)
