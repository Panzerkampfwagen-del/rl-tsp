"""OOD evaluation across all 5 city distributions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import numpy as np
import torch

from config import Config
from src.env import set_seed, sample_instances, DISTRIBUTIONS
from src.model import TSPModel
from src.evaluate import evaluate_model
from src.plotting import plot_shift_bar_chart, plot_all_distributions


def run_shift_analysis(
    checkpoint: str,
    cfg: Config,
    n_test: int = 1000,
    test_seed: int = 1234,
    out_dir: str = "figures",
) -> dict:
    set_seed(test_seed)
    os.makedirs(out_dir, exist_ok=True)

    model = TSPModel(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers, d_ff=cfg.d_ff,
        tanh_clip=cfg.tanh_clip,
    ).to(cfg.device)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device, weights_only=True))
    model.eval()

    results = {}
    rng = np.random.default_rng(test_seed)

    for dist in DISTRIBUTIONS:
        cities = sample_instances(dist, n_test, cfg.n_cities, device=cfg.device, rng=rng)
        metrics = evaluate_model(model, cities, batch_size=cfg.batch_size)
        method  = metrics["baseline_method"]
        results[dist] = {
            "gap_mean":       metrics["gap_vs_best_mean"],
            "gap_std":        metrics["gap_vs_best_std"],
            "model_len_mean": metrics["model_len_mean"],
            "model_len_std":  metrics["model_len_std"],
            "baseline_method": method,
        }
        print(f"{dist:10s}  gap={results[dist]['gap_mean']:+.2f}% ± {results[dist]['gap_std']:.2f}%  "
              f"tour={results[dist]['model_len_mean']:.4f}  baseline={method}")

    bar_path = os.path.join(out_dir, "shift_analysis.png")
    plot_shift_bar_chart(results, method, bar_path, n_cities=cfg.n_cities)
    print(f"Saved bar chart → {bar_path}")

    dist_path = os.path.join(out_dir, "distributions_overview.png")
    plot_all_distributions(n_per_dist=1, n_cities=cfg.n_cities, save_path=dist_path)
    print(f"Saved distributions overview → {dist_path}")

    json_path = os.path.join(out_dir, "shift_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON → {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_cities", type=int, default=20)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--test_seed", type=int, default=1234)
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    cfg = Config()
    cfg.n_cities = args.n_cities
    run_shift_analysis(args.checkpoint, cfg, args.n_test, args.test_seed, args.out_dir)
