"""Configuration for a paper-faithful CausalEPP implementation."""
from dataclasses import dataclass

@dataclass
class Config:
    dataset: str = "Ciao"
    data_dir: str = "data/Ciao"
    raw_file: str = "data/Ciao/ratings.csv"
    seed: int = 42
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    num_time_bins: int = 100
    high_popularity_quantile: float = 0.80
    embedding_dim: int = 64
    lightgcn_layers: int = 3
    mlp_hidden_dim: int = 32
    alpha: float = 0.5
    quality_loss_weight: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    epochs: int = 100
    eval_k: int = 20
    moving_average_window: int = 10
    item_forecast_horizon: float = 5.0
    user_forecast_horizon: float = 10.0
    device: str = "auto"
    output_dir: str = "outputs/Ciao"
