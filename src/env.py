"""TSP instance generation and tour evaluation."""

import torch
import numpy as np
from typing import Optional


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tour_length(cities: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """
    Compute tour length (returns to start).

    cities: (B, N, 2)
    tour:   (B, N) — indices, permutation of 0..N-1
    returns: (B,)
    """
    B, N, _ = cities.shape
    ordered = cities[torch.arange(B, device=cities.device).unsqueeze(1), tour]  # (B,N,2)
    shifted = torch.roll(ordered, -1, dims=1)
    return torch.norm(ordered - shifted, dim=2).sum(dim=1)


# --- instance generators ---

def sample_uniform(batch: int, n: int, device: str = "cpu",
                   rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Uniform random cities in [0,1]^2."""
    if rng is None:
        pts = np.random.rand(batch, n, 2).astype(np.float32)
    else:
        pts = rng.random((batch, n, 2)).astype(np.float32)
    return torch.from_numpy(pts).to(device)


def sample_clustered(batch: int, n: int, device: str = "cpu",
                     n_clusters: int = 3,
                     rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Cities drawn from a Gaussian mixture; clipped to [0,1]^2."""
    rs = rng if rng is not None else np.random.default_rng()
    pts = np.zeros((batch, n, 2), dtype=np.float32)
    for b in range(batch):
        k = rs.integers(2, n_clusters + 1)
        centers = rs.uniform(0.1, 0.9, size=(k, 2))
        for i in range(n):
            c = rs.integers(k)
            pts[b, i] = rs.normal(centers[c], 0.08, size=2)
    pts = np.clip(pts, 0.0, 1.0)
    return torch.from_numpy(pts).to(device)


def sample_grid(batch: int, n: int, device: str = "cpu",
                jitter: float = 0.02,
                rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Cities snapped to a regular grid with small jitter."""
    rs = rng if rng is not None else np.random.default_rng()
    side = int(np.ceil(np.sqrt(n)))
    g = np.linspace(0.05, 0.95, side)
    gx, gy = np.meshgrid(g, g)
    all_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (side*side, 2)
    pts = np.empty((batch, n, 2), dtype=np.float32)
    for b in range(batch):
        chosen_idx = rs.choice(len(all_pts), size=n, replace=False)
        pts[b] = all_pts[chosen_idx]  # (n, 2) — independent random sample per instance
    if jitter > 0:
        noise = rs.uniform(-jitter, jitter, size=(batch, n, 2)).astype(np.float32)
        pts = np.clip(pts + noise, 0.0, 1.0)
    return torch.from_numpy(pts).to(device)


def sample_circle(batch: int, n: int, device: str = "cpu",
                  noise: float = 0.02,
                  rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Cities on a noisy ring."""
    rs = rng if rng is not None else np.random.default_rng()
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.empty((batch, n, 2), dtype=np.float32)
    for b in range(batch):
        phase = rs.uniform(0, 2 * np.pi)          # independent rotation per instance
        radius = 0.4 + rs.normal(0, noise)        # independent radius per instance
        angles = base_angles + phase
        pts[b] = np.stack([0.5 + radius * np.cos(angles),
                           0.5 + radius * np.sin(angles)], axis=1)
    if noise > 0:
        pts += rs.normal(0, noise, size=(batch, n, 2)).astype(np.float32)
    pts = np.clip(pts, 0.0, 1.0)
    return torch.from_numpy(pts).to(device)


def sample_compact(batch: int, n: int, device: str = "cpu",
                   sub_size: float = 0.4,
                   rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Cities confined to a bottom-left sub-square of [0,1]^2."""
    if rng is None:
        pts = np.random.rand(batch, n, 2).astype(np.float32) * sub_size
    else:
        pts = rng.random((batch, n, 2)).astype(np.float32) * sub_size
    return torch.from_numpy(pts).to(device)


DISTRIBUTIONS = {
    "uniform":   sample_uniform,
    "clustered": sample_clustered,
    "grid":      sample_grid,
    "circle":    sample_circle,
    "compact":   sample_compact,
}


def sample_instances(dist: str, batch: int, n: int, device: str = "cpu",
                     rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {dist}")
    return DISTRIBUTIONS[dist](batch, n, device=device, rng=rng)


if __name__ == "__main__":
    # Quick hand-checkable 3-city test
    cities = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # (1,3,2)
    tour = torch.tensor([[0, 1, 2]])
    # edges: 0->1 = 1.0, 1->2 = sqrt(2), 2->0 = 1.0  => total = 2 + sqrt(2) ≈ 3.4142
    length = tour_length(cities, tour)
    expected = 2.0 + 2.0 ** 0.5
    print(f"3-city tour length: {length.item():.6f}  expected: {expected:.6f}  match: {abs(length.item()-expected)<1e-5}")
