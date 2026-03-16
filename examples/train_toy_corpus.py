from __future__ import annotations

from pathlib import Path

from a_small_numpy_word2vec.config import CorpusConfig, TrainConfig
from a_small_numpy_word2vec.preprocessing import read_text
from a_small_numpy_word2vec.training import train_word2vec


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_corpus_text = read_text(root / "datasets" / "toy_corpus.txt")

    model = train_word2vec(
        raw_corpus_text,
        corpus_config=CorpusConfig(min_count=1, subsample_t=0.0),
        train_config=TrainConfig(
            embedding_dim=32,
            max_context_window_size=2,
            negative_samples=4,
            learning_rate=0.05,
            epochs=200,
            seed=0,
        ),
    )

    for target_word in ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]:
        print(f"\ncluster near {target_word!r}:")
        for token, score in model.nearest_words(target_word, top_k=5):
            print(f"  {token:<12} {score:.4f}")


if __name__ == "__main__":
    main()