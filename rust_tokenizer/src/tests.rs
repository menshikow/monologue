// acsess private crates

use super::*;

#[test]
fn encode_returns_some_tokens() {
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.encode("hello world");
    assert!(
        !tokens.is_empty(),
        "encode should produce at least one token for non-empty input"
    );
}

#[test]
fn encode_batch_matches_individual_encode() {
    let tokenizer = Tokenizer::default();
    let inputs = vec!["hello".to_string(), "world".to_string()];

    let batch_tokens = tokenizer.encode_batch(inputs.clone());
    assert_eq!(batch_tokens.len(), inputs.len());

    for (i, text) in inputs.iter().enumerate() {
        let single = tokenizer.encode(text);
        assert_eq!(
            batch_tokens[i], single,
            "encode_batch result should match encode for input index {}",
            i
        );
    }
}

#[test]
fn train_core_increases_vocab_size() {
    let mut tokenizer = Tokenizer::default();
    let initial_vocab = tokenizer.vocab_size();

    // Small synthetic "corpus"
    let words = vec![
        Word::new("hello".bytes().map(|b| b as u32).collect()),
        Word::new("world".bytes().map(|b| b as u32).collect()),
        Word::new("hello world".bytes().map(|b| b as u32).collect()),
    ];
    let counts = vec![10, 8, 5];

    // Request a vocab larger than the base 256 bytes to force merges
    tokenizer.train_core(words, counts, 300);

    let new_vocab = tokenizer.vocab_size();
    assert!(
        new_vocab > initial_vocab,
        "vocab_size should increase after training; before={}, after={}",
        initial_vocab,
        new_vocab
    );
}
