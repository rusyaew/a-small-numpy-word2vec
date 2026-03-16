from __future__ import annotations

import numpy as np

from .config import CorpusConfig, TrainConfig
from .preprocessing import NegativeSampler, TrainingCorpus, iter_skipgram_pairs, prepare_corpus
from .model import (
    Word2Vec,
    apply_negative_sampling_update,
    init_embeddings,
    negative_sampling_step,
)

def train_word2vec(
        text: str,
        *,
        corpus_config: CorpusConfig,
        train_config: TrainConfig,
) -> Word2Vec:
    rng = np.random.default_rng(train_config.seed)
    corpus = prepare_corpus(text, corpus_config, rng)
    return train_on_corpus(corpus, train_config=train_config, rng=rng)


def train_on_corpus(
        corpus: TrainingCorpus,
        *,
        train_config: TrainConfig,
        rng: np.random.Generator,
) -> Word2Vec:
    vocab_size = len(corpus.vocabulary)
    if vocab_size == 0:
        raise ValueError("train: empty post-preprocessing vocabulary")
    if len(corpus.token_ids) < 2:
        raise ValueError("train: < 2 tokens post-preprocessing")

    emb_in, emb_out = init_embeddings(
        vocab_size=vocab_size,
        embedding_dim=train_config.embedding_dim,
        rng=rng,
    )
    neg_sampler = NegativeSampler.from_counts(corpus.vocabulary.counts)

    mean_losses: list[float] = []

    for epoch in range(train_config.epochs):
        loss_sum = 0.0
        n_steps = 0

        for center_id, outside_id in iter_skipgram_pairs(
                corpus.token_ids,
                max_context_window_size=train_config.max_context_window_size,
                rng=rng,
        ):
            neg_ids = neg_sampler.sample(
                train_config.negative_samples,
                rng,
                banned=outside_id,
            )

            step = negative_sampling_step(
                center_vec=emb_in[center_id],
                outside_vec=emb_out[outside_id],
                neg_vecs=emb_out[neg_ids],
            )
            apply_negative_sampling_update(
                emb_in=emb_in,
                emb_out=emb_out,
                center_id=center_id,
                outside_id=outside_id,
                neg_ids=neg_ids,
                step=step,
                learning_rate=train_config.learning_rate,
            )

            loss_sum += float(step.loss)
            n_steps += 1

        mean_loss = loss_sum / max(n_steps, 1)
        mean_losses.append(mean_loss)
        print(f"train: epoch={epoch + 1}/{train_config.epochs} mean_loss={mean_loss:.4f}")

    return Word2Vec(
        vocabulary=corpus.vocabulary,
        emb_in=emb_in,
        emb_out=emb_out,
        mean_losses=mean_losses,
    )