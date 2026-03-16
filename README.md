### Usage
Train a small model:

```python
from a_small_numpy_word2vec import CorpusConfig, TrainConfig, train_word2vec

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

print(model.embeddings.shape)
print(model.mean_losses)
print(model.nearest_words("lantern", top_k=3))
```

### Example
Example `examples/train_toy_corpus.py ` for testing on handwritten toy corpus
```text 
train: epoch=1/200 mean_loss=3.4656
train: epoch=2/200 mean_loss=3.4632
...
train: epoch=200/200 mean_loss=1.7252

cluster near 'king':
  man          0.6854
  male         0.6248
  lives        0.5685
  the          0.5477
  father       0.5352

cluster near 'queen':
  lives        0.7055
  rules        0.6954
  the          0.6285
  a            0.6129
  in           0.5464

...

cluster near 'girl':
  young        0.7683
  mother       0.6927
  is           0.6720
  female       0.6513
  male         0.5961
```

### Testing
Smoke tests via `pytest`:
```text
============================================================================ test session starts ============================================================================
platform linux -- Python 3.11.0rc1, pytest-9.0.2, pluggy-1.6.0 -- /home/gleb/IdeaProjects/a-small-numpy-word2vec/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/gleb/IdeaProjects/a-small-numpy-word2vec
configfile: pyproject.toml
testpaths: tests
collected 2 items                                                                                                                                                           

tests/test.py::test_tokenize_and_prepare_corpus PASSED                                                                                                                [ 50%]
tests/test.py::test_training_runs PASSED                                                                                                                              [100%]

============================================================================= 2 passed in 0.18s =============================================================================
```

### Miscellaneous

I also used style checker, linter (Ruff) and PR (GitHub Actions CI from `dev/nika` to `main` with passed tests) to mimic the production environment.
