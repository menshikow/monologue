import pytest


def test_encode_basic():
    """
    Basic smoke test for the Python-facing Tokenizer.
    Skips automatically if the `rust_tokenizer` extension module
    is not installed (e.g. until `maturin develop` has been run).
    """
    rust_tokenizer = pytest.importorskip("rust_tokenizer")

    tok = rust_tokenizer.Tokenizer()
    tokens = tok.encode("hello world")

    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0


def test_encode_batch_consistency():
    rust_tokenizer = pytest.importorskip("rust_tokenizer")

    tok = rust_tokenizer.Tokenizer()
    texts = ["hello", "world"]

    batch = tok.encode_batch(texts)

    assert isinstance(batch, list)
    assert len(batch) == len(texts)

    for i, text in enumerate(texts):
        single = tok.encode(text)
        assert batch[i] == single


def test_vocab_size_and_special_tokens():
    rust_tokenizer = pytest.importorskip("rust_tokenizer")

    tok = rust_tokenizer.Tokenizer()
    base_vocab = tok.vocab_size()

    # Register a special token and ensure encode picks it up
    tok.register_special_token("<SPECIAL>", base_vocab + 1)

    tokens = tok.encode("<SPECIAL>")
    assert tokens == [base_vocab + 1]
