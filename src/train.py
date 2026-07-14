"""REINFORCE training loop with greedy rollout baseline."""

import os
import sys
import csv
import argparse
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config, small_config
from src.env import set_seed, sample_uniform, tour_length
from src.model import TSPModel
from src.baseline import GreedyRolloutBaseline


def print_gpu_info(device: str) -> None:
    if "cuda" in device and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"GPU: {props.name}  VRAM: {vram_gb:.1f}GB  "
              f"CUDA: {torch.version.cuda}  PyTorch: {torch.__version__}")
    else:
        print(f"Running on CPU.  PyTorch: {torch.__version__}")


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    print_gpu_info(cfg.device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model = TSPModel(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers, d_ff=cfg.d_ff,
        tanh_clip=cfg.tanh_clip,
    ).to(cfg.device)

    baseline = GreedyRolloutBaseline(
        model, cfg.n_cities, cfg.device,
        val_size=cfg.val_size, p_threshold=cfg.baseline_update_pvalue,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    orig_model = model
    if cfg.device != "cpu":
        torch.set_float32_matmul_precision("high")
        try:
            import warnings, logging
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.compile(model)
            print("torch.compile enabled (TF32 on)")
        except Exception as e:
            print(f"torch.compile failed ({e}); running in eager mode.")

    log_path = os.path.join(cfg.checkpoint_dir, cfg.log_file)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "avg_loss", "avg_tour_len", "bl_updated", "p_value", "elapsed_s"])

    wandb_run = None
    if cfg.use_wandb:
        import wandb
        wandb_run = wandb.init(project="rl-tsp", config=vars(cfg))

    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_len  = 0.0

        for _ in range(cfg.steps_per_epoch):
            cities = sample_uniform(cfg.batch_size, cfg.n_cities, device=cfg.device)
            tour, log_probs = model(cities)
            lengths = tour_length(cities, tour)

            with torch.no_grad():
                baseline_len = baseline.get_baseline(cities)

            advantage = lengths - baseline_len
            loss = (advantage * log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len  += lengths.mean().item()

        avg_loss = epoch_loss / cfg.steps_per_epoch
        avg_len  = epoch_len  / cfg.steps_per_epoch
        bl_updated, p_val = baseline.update_if_better(val_seed=cfg.seed + epoch)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{cfg.n_epochs}  loss={avg_loss:+.4f}  "
              f"tour_len={avg_len:.4f}  bl_updated={bl_updated}  "
              f"p={p_val:.4f}  t={elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss, avg_len, bl_updated, p_val, elapsed])

        if wandb_run:
            wandb_run.log({"loss": avg_loss, "tour_len": avg_len, "epoch": epoch})

        if epoch % 10 == 0 or epoch == cfg.n_epochs:
            ckpt = os.path.join(cfg.checkpoint_dir, f"model_ep{epoch:04d}.pt")
            torch.save(orig_model.state_dict(), ckpt)

    if wandb_run:
        wandb_run.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="smoke test: TSP-10, 5 epochs")
    parser.add_argument("--n_cities", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = small_config() if args.small else Config()
    if args.n_cities:       cfg.n_cities       = args.n_cities
    if args.batch_size:     cfg.batch_size     = args.batch_size
    if args.n_epochs:       cfg.n_epochs       = args.n_epochs
    if args.steps_per_epoch: cfg.steps_per_epoch = args.steps_per_epoch
    if args.wandb:          cfg.use_wandb      = True

    train(cfg)
