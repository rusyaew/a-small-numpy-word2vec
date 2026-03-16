from __future__ import annotations

import numpy as np

from a_small_numpy_word2vec.config import CorpusConfig, TrainConfig
from a_small_numpy_word2vec.preprocessing import prepare_corpus, tokenize
from a_small_numpy_word2vec.training import train_word2vec


def test_tokenize_and_prepare_corpus() -> None:
    rng = np.random.default_rng(0)

    text = (
        "If narrative development implies the progression of chronological time, "
        "history and its progression is determined by influx and outflow of petroleum"
    )

    tokens = tokenize(text, lowercase=True)

    assert tokens == [
        "if",
        "narrative",
        "development",
        "implies",
        "the",
        "progression",
        "of",
        "chronological",
        "time",
        ",",
        "history",
        "and",
        "its",
        "progression",
        "is",
        "determined",
        "by",
        "influx",
        "and",
        "outflow",
        "of",
        "petroleum",
    ]

    corpus = prepare_corpus(
        "velvet crown velvet lantern silver crown",
        CorpusConfig(min_count=1, subsample_t=0.0),
        rng,
    )

    assert len(corpus.vocabulary) == 4
    assert corpus.token_ids.dtype == np.int64
    assert corpus.vocabulary.counts.dtype == np.int64


def test_training_runs() -> None:
    text = (
        "amber comet amber comet "
        "harbor lantern harbor lantern "
        "copper bridge velvet lantern"
    )

    model = train_word2vec(
        text,
        corpus_config=CorpusConfig(min_count=1, subsample_t=0.0),
        train_config=TrainConfig(
            embedding_dim=8,
            max_context_window_size=2,
            negative_samples=2,
            learning_rate=0.05,
            epochs=5,
            seed=0,
        ),
    )

    assert len(model.vocabulary) > 0
    assert model.embeddings.shape[1] == 8
    assert len(model.mean_losses) == 5
    assert np.all(np.isfinite(model.embeddings))