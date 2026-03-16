from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CorpusConfig
from .typing_ import FloatArray, IntArray

_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?|\d+|[^\w\s]")


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    counts: IntArray

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> IntArray:
        ids = [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        return np.asarray(ids, dtype=np.int64)

    def decode(self, ids: IntArray) -> list[str]:
        return [self.id_to_token[int(token_id)] for token_id in ids]

    def token(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def count(self, token_id: int) -> int:
        return int(self.counts[token_id])


@dataclass(frozen=True)
class TrainingCorpus:
    vocabulary: Vocabulary
    token_ids: IntArray


@dataclass(frozen=True)
class NegativeSampler:
    probs: FloatArray

    @classmethod
    def from_counts(cls, counts: IntArray, power: float = 0.75) -> NegativeSampler:
        adjusted_counts = np.power(counts.astype(np.float64), power)
        probs = adjusted_counts / adjusted_counts.sum()
        return cls(probs=probs.astype(np.float32))

    def sample(self, k: int, rng: np.random.Generator, banned: int | None = None) -> IntArray:
        if banned is None:
            return rng.choice(len(self.probs), size=k, replace=True, p=self.probs).astype(
                np.int64,
                copy=False
            )

        sampled: list[int] = []
        while len(sampled) < k:
            remaining_samples = k - len(sampled)
            batch = rng.choice(
                len(self.probs),
                size=remaining_samples,
                replace=True,
                p=self.probs
            )
            sampled.extend(int(token_id) for token_id in batch if int(token_id) != banned)
        return np.asarray(sampled, dtype=np.int64)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def tokenize(
        text: str,
        *,
        lowercase: bool = True,
        pattern: re.Pattern[str] = _TOKEN_PATTERN,
) -> list[str]:
    source = text.lower() if lowercase else text
    return pattern.findall(source)


def build_vocabulary(tokens: list[str], config: CorpusConfig) -> Vocabulary:
    freqs = Counter(tokens) # HashMap counter working in amortized O(n) time for large datasets
    kept = [(token, count) for token, count in freqs.items() if count >= config.min_count]
    kept.sort(key=lambda item: (-item[1], item[0]))

    id_to_token = [token for token, _ in kept]
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    counts = np.asarray([count for _, count in kept], dtype=np.int64)

    return Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        counts=counts
    )


def subsample_token_ids(
        token_ids: IntArray,
        counts: IntArray,
        threshold: float,
        rng: np.random.Generator,
) -> IntArray:
    if len(token_ids) == 0 or threshold <= 0.0:
        return token_ids.copy()

    total = float(np.sum(counts, dtype=np.int64))
    rel_freq = counts.astype(np.float64) / total
    keep_prob = np.minimum(1.0, np.sqrt(threshold / rel_freq))

    draws = rng.random(len(token_ids))
    keep = draws < keep_prob[token_ids]
    return token_ids[keep].astype(np.int64, copy=False)


def prepare_corpus(
        text: str,
        config: CorpusConfig,
        rng: np.random.Generator,
        *,
        pattern: re.Pattern[str] = _TOKEN_PATTERN,
) -> TrainingCorpus:
    tokens = tokenize(text, lowercase=config.lowercase, pattern=pattern)
    vocab = build_vocabulary(tokens, config)
    token_ids = vocab.encode(tokens)

    subsampled_ids = subsample_token_ids(token_ids, vocab.counts, config.subsample_t, rng)
    if len(subsampled_ids) >= 2:
        token_ids = subsampled_ids

    return TrainingCorpus(vocabulary=vocab, token_ids=token_ids)


def iter_skipgram_pairs(
        token_ids: IntArray,
        max_context_window_size: int,
        rng: np.random.Generator,
) -> Iterator[tuple[int, int]]:
    if max_context_window_size < 1:
        raise ValueError("preprocessing: max_context_window_size must be > 0")

    amount_of_tokens = len(token_ids)
    for center_pos, center_id in enumerate(token_ids):
        dynamic_context_window = int(rng.integers(1, max_context_window_size + 1))
        lo = max(0, center_pos - dynamic_context_window)
        hi = min(amount_of_tokens, center_pos + dynamic_context_window + 1)

        for outside_pos in range(lo, hi):
            if outside_pos == center_pos:
                continue
            yield int(center_id), int(token_ids[outside_pos])