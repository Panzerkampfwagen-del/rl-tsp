"""Nearest-neighbor, 2-opt, OR-Tools, and random-tour baselines."""

import torch
import numpy as np
from typing import Optional

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


def nearest_neighbor_tour(cities_np: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor starting from city 0. cities_np: (N, 2)."""
    N = len(cities_np)
    visited = np.zeros(N, dtype=bool)
    tour = np.empty(N, dtype=np.int64)
    cur = 0
    visited[cur] = True
    tour[0] = cur
    for step in range(1, N):
        dists = np.linalg.norm(cities_np - cities_np[cur], axis=1)
        dists[visited] = np.inf
        cur = int(np.argmin(dists))
        visited[cur] = True
        tour[step] = cur
    return tour


def two_opt_improve(cities_np: np.ndarray, tour: np.ndarray,
                    max_iter: int = 1000) -> np.ndarray:
    """2-opt local search. Returns improved tour."""
    N = len(tour)
    tour = tour.copy()
    dist = np.linalg.norm(cities_np[:, None] - cities_np[None, :], axis=-1)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(N - 1):
            for j in range(i + 2, N):
                a, b = tour[i], tour[(i + 1) % N]
                c, d = tour[j], tour[(j + 1) % N]
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-8:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _ortools_solve(cities_np: np.ndarray) -> Optional[np.ndarray]:
    N = len(cities_np)
    dist_matrix = (np.linalg.norm(
        cities_np[:, None, :] - cities_np[None, :, :], axis=2
    ) * 1e6).astype(int)

    manager = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)])

    idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    tour = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return np.array(tour, dtype=np.int64)


def best_tour(cities_np: np.ndarray, label: bool = False):
    """
    OR-Tools if available, else nearest-neighbor + 2-opt.
    Returns (tour_np, method_name) if label=True, else just tour_np.
    """
    if _HAS_ORTOOLS:
        tour = _ortools_solve(cities_np)
        method = "OR-Tools"
        if tour is None:
            tour = two_opt_improve(cities_np, nearest_neighbor_tour(cities_np))
            method = "2-opt"
    else:
        tour = two_opt_improve(cities_np, nearest_neighbor_tour(cities_np))
        method = "2-opt"
    return (tour, method) if label else tour


def batch_nn_tours(cities: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor for a batch. cities: (B,N,2) → tours (B,N)."""
    B = cities.shape[0]
    arr = cities.cpu().numpy()
    tours = np.stack([nearest_neighbor_tour(arr[b]) for b in range(B)])
    return torch.from_numpy(tours).to(cities.device)


def batch_best_tours(cities: torch.Tensor) -> tuple:
    """Best heuristic for a batch. Returns (tours tensor, method_name)."""
    B = cities.shape[0]
    arr = cities.cpu().numpy()
    results = [best_tour(arr[b], label=True) for b in range(B)]
    tours = np.stack([r[0] for r in results])
    methods = [r[1] for r in results]
    method = methods[0] if len(set(methods)) == 1 else "2-opt"
    return torch.from_numpy(tours).to(cities.device), method


def random_tours(cities: torch.Tensor) -> torch.Tensor:
    """Random permutation tours. cities: (B,N,2) → (B,N)."""
    B, N, _ = cities.shape
    return torch.stack([torch.randperm(N) for _ in range(B)]).to(cities.device)


if __name__ == "__main__":
    print(f"OR-Tools available: {_HAS_ORTOOLS}")
    np.random.seed(0)
    c = np.random.rand(10, 2)
    nn_t = nearest_neighbor_tour(c)
    t2 = two_opt_improve(c, nn_t)
    lengths = [sum(np.linalg.norm(c[t[i]] - c[t[(i+1) % 10]]) for i in range(10))
               for t in [nn_t, t2]]
    print(f"NN length: {lengths[0]:.4f}  2-opt length: {lengths[1]:.4f}")
