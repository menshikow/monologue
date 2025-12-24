// benches/tokenizer_bench.rs
// Benchmarks for tokenizer performance

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rust_tokenizer::Tokenizer;
use std::collections::HashMap;

fn create_trained_tokenizer() -> Tokenizer {
    let mut tok = Tokenizer::new().unwrap();

    // Add some common English merges
    let mut merges = HashMap::new();
    merges.insert((116, 104), 256); // "th"
    merges.insert((105, 110), 257); // "in"
    merges.insert((101, 114), 258); // "er"
    merges.insert((97, 110), 259); // "an"
    merges.insert((111, 110), 260); // "on"

    tok.load_merges(merges);
    tok
}

/// Benchmark encoding cost as a function of input length.
fn bench_encode_by_length(c: &mut Criterion) {
    let tok = create_trained_tokenizer();

    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(10);

    let samples: Vec<(&str, &str)> = vec![
        ("short", "hello"),
        ("medium", "The quick brown fox jumps over the lazy dog"),
        ("long", &long_text),
    ];

    let mut group = c.benchmark_group("encode_by_length");

    for (label, text) in samples {
        group.bench_with_input(BenchmarkId::new("encode", label), text, |b, text| {
            b.iter(|| tok.encode(black_box(text)))
        });
    }

    group.finish();
}

/// Helper that benchmarks pure encoding cost for batches without
/// re-cloning the input vector on every iteration.
fn encode_batch_pure(tok: &Tokenizer, texts: &[String]) -> Vec<Vec<u32>> {
    texts.iter().map(|t| tok.encode(t)).collect()
}

fn bench_batch_encode_small(c: &mut Criterion) {
    let tok = create_trained_tokenizer();

    let texts: Vec<String> = (0..10)
        .map(|i| format!("Sample text number {}", i))
        .collect();

    c.bench_function("batch_encode_small_pure", |b| {
        b.iter(|| encode_batch_pure(&tok, black_box(&texts)))
    });
}

fn bench_batch_encode_large(c: &mut Criterion) {
    let tok = create_trained_tokenizer();

    let texts: Vec<String> = (0..200)
        .map(|i| format!("Sample text number {}", i))
        .collect();

    c.bench_function("batch_encode_large_pure", |b| {
        b.iter(|| encode_batch_pure(&tok, black_box(&texts)))
    });
}

fn bench_register_special_token(c: &mut Criterion) {
    // Measures cold-path cost: constructing a tokenizer + registering a token.
    c.bench_function("new_plus_register_special_token", |b| {
        b.iter(|| {
            let mut tok = Tokenizer::new().unwrap();
            tok.register_special_token(black_box("<PAD>".to_string()), 50000);
        })
    });
}

fn bench_get_merges(c: &mut Criterion) {
    let tok = create_trained_tokenizer();

    c.bench_function("get_merges", |b| b.iter(|| tok.get_merges()));
}

fn bench_load_merges(c: &mut Criterion) {
    let merges: HashMap<(u32, u32), u32> = (0..1000)
        .map(|i| ((i % 256, (i + 1) % 256), 256 + i))
        .collect();

    // Measures cold-path cost: constructing a tokenizer + loading 1k merges.
    c.bench_function("new_plus_load_merges_1000", |b| {
        b.iter(|| {
            let mut tok = Tokenizer::new().unwrap();
            tok.load_merges(black_box(merges.clone()));
        })
    });
}

criterion_group!(
    benches,
    bench_encode_by_length,
    bench_batch_encode_small,
    bench_batch_encode_large,
    bench_register_special_token,
    bench_get_merges,
    bench_load_merges,
);

criterion_main!(benches);
