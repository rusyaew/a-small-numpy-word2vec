from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusConfig:
    min_count: int = 1
    subsample_t: float = 1e-5
    lowercase: bool = True


@dataclass(frozen=True)
class TrainConfig:
    embedding_dim: int = 64
    max_context_window_size: int = 5
    negative_samples: int = 5
    learning_rate: float = 0.025
    epochs: int = 3
    seed: int = 0
    report_every: int = 10_000