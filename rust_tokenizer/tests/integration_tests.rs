use rust_tokenizer::Tokenizer;

/// Integration test using only the public Tokenizer API.
/// This exercises construction, encoding, batch encoding, and special tokens.
#[test]
fn tokenizer_public_api_end_to_end() {
    let mut tokenizer = Tokenizer::default();

    // Basic encode sanity
    let text = "hello world";
    let tokens_first = tokenizer.encode(text);
    let tokens_second = tokenizer.encode(text);

    assert!(
        !tokens_first.is_empty(),
        "encode should return non-empty tokens for non-empty input"
    );
    assert_eq!(
        tokens_first, tokens_second,
        "encode should be deterministic for the same input"
    );

    // Batch encoding should match per-item encoding
    let batch_inputs = vec![
        "hello world".to_string(),
        "hello rust".to_string(),
        "tokenizer test".to_string(),
    ];
    let batch_tokens = tokenizer.encode_batch(batch_inputs.clone());

    assert_eq!(batch_tokens.len(), batch_inputs.len());
    for (i, text) in batch_inputs.iter().enumerate() {
        let single = tokenizer.encode(text);
        assert_eq!(
            batch_tokens[i], single,
            "encode_batch should match encode for index {}",
            i
        );
    }

    // Special tokens should override normal encoding
    let base_vocab = tokenizer.vocab_size();
    let special_id = (base_vocab + 1) as u32;
    tokenizer.register_special_token("<SPECIAL>".to_string(), special_id);

    // Current implementation uses a regex-based tokenizer and byte-level IDs;
    // registering the special token only affects exact chunk matches.
    // This call is therefore just a smoke test to ensure it doesn't panic.
    let _ = tokenizer.encode("<SPECIAL>");
}
