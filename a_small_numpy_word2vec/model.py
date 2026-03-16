from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evals import nearest_words as _nearest_words
from .preprocessing import Vocabulary
from .typing_ import FloatArray, IntArray


@dataclass(frozen=True)
class Word2Vec:
    vocabulary: Vocabulary
    emb_in: FloatArray
    emb_out: FloatArray
    mean_losses: list[float]

    @property
    def embeddings(self) -> FloatArray:
        return average_embeddings(self.emb_in, self.emb_out)

    def nearest_words(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        return _nearest_words(word, self.embeddings, self.vocabulary, top_k=top_k)

@dataclass(frozen=True)
class NegativeSamplingStep:
    loss: np.float32
    grad_center: FloatArray
    grad_outside: FloatArray
    grad_negs: FloatArray


def init_embeddings(
        vocab_size: int,
        embedding_dim: int,
        rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    init_bound = np.float32(0.5 / max(embedding_dim, 1))
    emb_in = rng.uniform(
        -init_bound,
        init_bound,
        size=(vocab_size, embedding_dim)
    ).astype(np.float32)
    emb_out = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    return emb_in, emb_out


def sigmoid(x: FloatArray | np.float32 | float) -> FloatArray | np.float32:
    x_arr = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x_arr)

    pos = x_arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos], dtype=np.float32))

    exp_x = np.exp(x_arr[~pos], dtype=np.float32)
    out[~pos] = exp_x / (1.0 + exp_x)

    if np.isscalar(x):
        return np.float32(out.item())
    return out


def negative_sampling_step(
        center_vec: FloatArray,
        outside_vec: FloatArray,
        neg_vecs: FloatArray,
) -> NegativeSamplingStep:
    score_pos = np.float32(outside_vec @ center_vec)
    score_neg = neg_vecs @ center_vec

    prob_pos = np.float32(sigmoid(score_pos))
    prob_neg = np.asarray(sigmoid(score_neg), dtype=np.float32)

    eps = np.float32(1e-8)
    loss = np.float32(
        -np.log(prob_pos + eps)
        - np.sum(np.log(1.0 - prob_neg + eps), dtype=np.float32)
    )

    pos_coeff = np.float32(prob_pos - 1.0)
    neg_coeffs = prob_neg

    grad_center = pos_coeff * outside_vec
    grad_center += np.sum(neg_coeffs[:, None] * neg_vecs, axis=0, dtype=np.float32)

    grad_outside = pos_coeff * center_vec
    grad_negs = neg_coeffs[:, None] * center_vec[None, :]

    return NegativeSamplingStep(
        loss=loss,
        grad_center=np.asarray(grad_center, dtype=np.float32),
        grad_outside=np.asarray(grad_outside, dtype=np.float32),
        grad_negs=np.asarray(grad_negs, dtype=np.float32),
    )


def apply_negative_sampling_update(
        emb_in: FloatArray,
        emb_out: FloatArray,
        center_id: int,
        outside_id: int,
        neg_ids: IntArray,
        step: NegativeSamplingStep,
        learning_rate: float,
) -> None:
    lr = np.float32(learning_rate)
    emb_in[center_id] -= lr * step.grad_center
    emb_out[outside_id] -= lr * step.grad_outside
    np.add.at(emb_out, neg_ids, -lr * step.grad_negs)


def average_embeddings(emb_in: FloatArray, emb_out: FloatArray) -> FloatArray:
    return ((emb_in + emb_out) / np.float32(2.0)).astype(np.float32, copy=False)