from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # Problem
    n_cities: int = 20
    seed: int = 42

    # Architecture
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_ff: int = 512
    tanh_clip: float = 10.0

    # Training
    batch_size: int = 256
    n_epochs: int = 100
    steps_per_epoch: int = 500
    lr: float = 1e-4
    grad_clip: float = 1.0
    baseline_update_pvalue: float = 0.05

    # Evaluation
    val_size: int = 1000
    test_size: int = 1000
    test_seed: int = 1234

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    use_wandb: bool = False
    checkpoint_dir: str = "checkpoints"
    figures_dir: str = "figures"
    log_file: str = "training_log.csv"


def small_config() -> Config:
    """TSP-10 config for smoke tests / OOM fallback."""
    c = Config()
    c.n_cities = 10
    c.batch_size = 128
    c.n_epochs = 5
    c.steps_per_epoch = 100
    return c
