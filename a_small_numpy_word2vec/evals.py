from __future__ import annotations

import numpy as np

from .preprocessing import Vocabulary
from .typing_ import FloatArray


def l2_normalized(embeddings: FloatArray, *, eps: float = 1e-12) -> FloatArray:
    row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, eps)
    return (embeddings / row_norms).astype(np.float32, copy=False)


def nearest_words(
        target_word: str,
        embeddings: FloatArray,
        vocabulary: Vocabulary,
        *,
        top_k: int = 10,
) -> list[tuple[str, float]]:
    if target_word not in vocabulary.token_to_id:
        raise KeyError(f"evals: unknown word \'{target_word}\'")

    normalized_embeddings = l2_normalized(embeddings)
    target_word_id = vocabulary.token_to_id[target_word]
    target_word_vec = normalized_embeddings[target_word_id]

    cosine_similarities = normalized_embeddings @ target_word_vec
    cosine_similarities[target_word_id] = -np.inf

    nearest_ids = np.argsort(cosine_similarities)[-top_k:][::-1]
    return [(vocabulary.token(int(token_id)), float(cosine_similarities[token_id])) for token_id in nearest_ids]