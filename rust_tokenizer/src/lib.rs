use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;

const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);

// Optimized chunk size for parallel processing
const PARALLEL_CHUNK_SIZE: usize = 256;
const MIN_PARALLEL_WORK: usize = 1000;

/// Represents a single word/chunk being processed.
/// Optimized with inline hints and better memory layout.
#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline(always)]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline(always)]
    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Highly optimized merge function with reduced allocations
    #[inline]
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();

        if n < 2 {
            return Vec::new();
        }

        // Pre-allocate with exact upper bounds
        let mut out = Vec::with_capacity((n + 1) / 2);
        let mut deltas = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            // Check for merge opportunity
            if i + 1 < n
                && unsafe { *self.ids.get_unchecked(i) } == a
                && unsafe { *self.ids.get_unchecked(i + 1) } == b
            {
                // Left neighbor update
                if let Some(&prev) = out.last() {
                    deltas.push(((prev, a), -1));
                    deltas.push(((prev, new_id), 1));
                }

                // Current pair removal
                deltas.push(((a, b), -1));

                // Right neighbor update
                if i + 2 < n {
                    let next = unsafe { *self.ids.get_unchecked(i + 2) };
                    deltas.push(((b, next), -1));
                    deltas.push(((new_id, next), 1));
                }

                out.push(new_id);
                i += 2;
            } else {
                out.push(unsafe { *self.ids.get_unchecked(i) });
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

/// Priority queue job with optimized comparison
#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.count.cmp(&other.count) {
            Ordering::Equal => other.pair.cmp(&self.pair),
            other => other,
        }
    }
}

// Main tokenizer class
#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Arc<Regex>,
    pub special_tokens: StdHashMap<String, u32>,
}

impl Tokenizer {
    /// Optimized parallel pair counting with adaptive chunking
    fn count_pairs_parallel(
        words: &[Word],
        counts: &[i32],
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        // Skip parallelization for small inputs
        if words.len() < MIN_PARALLEL_WORK {
            return Self::count_pairs_sequential(words, counts);
        }

        // Adaptive chunk size based on work size and thread count
        let num_threads = rayon::current_num_threads();
        let chunk_size = (words.len() / (num_threads * 4)).max(PARALLEL_CHUNK_SIZE);

        words
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;

                // Pre-size hash maps based on expected load
                let mut local_pc = AHashMap::with_capacity(chunk.len() * 2);
                let mut local_wtu = AHashMap::with_capacity(chunk.len() * 2);

                for (offset, w) in chunk.iter().enumerate() {
                    let i = base_idx + offset;
                    let count = unsafe { *counts.get_unchecked(i) };

                    if w.ids.len() >= 2 && count != 0 {
                        for pair in w.pairs() {
                            *local_pc.entry(pair).or_insert(0) += count;
                            local_wtu
                                .entry(pair)
                                .or_insert_with(AHashSet::new)
                                .insert(i);
                        }
                    }
                }
                (local_pc, local_wtu)
            })
            .reduce(
                || {
                    (
                        AHashMap::with_capacity(10000),
                        AHashMap::with_capacity(10000),
                    )
                },
                |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                    for (k, v) in pc {
                        *acc_pc.entry(k).or_insert(0) += v;
                    }
                    for (k, s) in wtu {
                        acc_wtu.entry(k).or_insert_with(AHashSet::new).extend(s);
                    }
                    (acc_pc, acc_wtu)
                },
            )
    }

    /// Sequential version for small inputs avoiding parallelization overhead
    #[inline]
    fn count_pairs_sequential(
        words: &[Word],
        counts: &[i32],
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        let mut pair_counts = AHashMap::with_capacity(words.len() * 2);
        let mut where_to_update = AHashMap::with_capacity(words.len() * 2);

        for (i, w) in words.iter().enumerate() {
            if w.ids.len() >= 2 && counts[i] != 0 {
                for pair in w.pairs() {
                    *pair_counts.entry(pair).or_insert(0) += counts[i];
                    where_to_update
                        .entry(pair)
                        .or_insert_with(AHashSet::new)
                        .insert(i);
                }
            }
        }

        (pair_counts, where_to_update)
    }

    /// Core BPE training loop
    fn train_core(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size >= 256, "vocab_size must be >= 256");
        let num_merges = vocab_size - 256;

        // Initial pair counting
        let (mut pair_counts, mut where_to_update) = Self::count_pairs_parallel(&words, &counts);

        // Build priority queue
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            if let Some(&c) = pair_counts.get(&pair) {
                if c > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: c as u64,
                        pos,
                    });
                }
            }
        }

        // Pre-allocate merge storage
        self.merges.reserve(num_merges as usize);
        let mut merges_done = 0;

        // Reusable buffer for local updates (avoid repeated allocations)
        let mut local_updates = AHashMap::with_capacity(1000);

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else { break };

            // Lazy staleness check
            let current = pair_counts.get(&top.pair).copied().unwrap_or(0);
            if top.count != current as u64 {
                if current > 0 {
                    top.count = current as u64;
                    heap.push(top);
                }
                continue;
            }

            // Record merge
            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            // Clear and reuse local_updates buffer
            local_updates.clear();

            // Update affected words
            for &word_idx in &top.pos {
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                let word_count = counts[word_idx];

                for (pair, delta) in changes {
                    let total_change = delta * word_count;
                    *pair_counts.entry(pair).or_insert(0) += total_change;

                    if delta > 0 {
                        local_updates
                            .entry(pair)
                            .or_insert_with(AHashSet::new)
                            .insert(word_idx);
                    }
                }
            }

            // Push updated pairs to heap
            for (pair, pos) in local_updates.drain() {
                if let Some(&cnt) = pair_counts.get(&pair) {
                    if cnt > 0 {
                        heap.push(MergeJob {
                            pair,
                            count: cnt as u64,
                            pos,
                        });
                    }
                }
            }

            merges_done += 1;
        }
    }
}

// python interface
#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new() -> PyResult<Self> {
        let compiled_pattern = Regex::new(GPT4_PATTERN).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to compile regex: {}",
                e
            ))
        })?;

        Ok(Self {
            merges: StdHashMap::with_capacity(50000),
            pattern: GPT4_PATTERN.to_string(),
            compiled_pattern: Arc::new(compiled_pattern),
            special_tokens: StdHashMap::new(),
        })
    }

    pub fn register_special_token(&mut self, token: String, id: u32) {
        self.special_tokens.insert(token, id);
    }

    /// Main training entry point - heavily optimized for throughput
    #[pyo3(signature = (iterator, vocab_size, buffer_size=10_000))]
    pub fn train_from_iterator(
        &mut self,
        py: Python<'_>,
        iterator: &Bound<'_, PyAny>,
        vocab_size: u32,
        buffer_size: usize,
    ) -> PyResult<()> {
        let py_iter = iterator.iter()?;

        // Pre-sized for large datasets
        let mut global_counts = AHashMap::with_capacity(200_000);
        let mut buffer = Vec::with_capacity(buffer_size);

        // Clone Arc for parallel use
        let pattern = Arc::clone(&self.compiled_pattern);

        loop {
            buffer.clear();

            // Fill buffer (holds GIL)
            let mut exhausted = false;
            for _ in 0..buffer_size {
                match py_iter.next() {
                    Some(Ok(item)) => buffer.push(item.extract::<String>()?),
                    Some(Err(e)) => return Err(e),
                    None => {
                        exhausted = true;
                        break;
                    }
                }
            }

            if buffer.is_empty() {
                break;
            }

            // Process batch in parallel (releases GIL)
            let batch_counts = py.allow_threads(|| {
                buffer
                    .par_iter()
                    .map(|text| {
                        let mut local_map = AHashMap::with_capacity(128);

                        for m in pattern.find_iter(text) {
                            if let Ok(m) = m {
                                *local_map
                                    .entry(CompactString::from(m.as_str()))
                                    .or_insert(0) += 1;
                            }
                        }
                        local_map
                    })
                    .reduce(
                        || AHashMap::with_capacity(2048),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_insert(0) += v;
                            }
                            a
                        },
                    )
            });

            // Merge into global counts
            for (k, v) in batch_counts {
                *global_counts.entry(k).or_insert(0) += v;
            }

            if exhausted {
                break;
            }
        }

        // Convert to training format
        let capacity = global_counts.len();
        let mut words = Vec::with_capacity(capacity);
        let mut counts_vec = Vec::with_capacity(capacity);

        for (token_str, count) in global_counts {
            let ids: Vec<u32> = token_str.bytes().map(|b| b as u32).collect();
            words.push(Word::new(ids));
            counts_vec.push(count);
        }

        // Run BPE
        self.train_core(words, counts_vec, vocab_size);
        Ok(())
    }

    /// Optimized encoding with better merge selection
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::with_capacity(text.len() / 4);

        for m in self.compiled_pattern.find_iter(text) {
            if let Ok(m) = m {
                let chunk = m.as_str();

                // Fast path for special tokens
                if let Some(&id) = self.special_tokens.get(chunk) {
                    result.push(id);
                    continue;
                }

                // Convert to token IDs
                let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

                // Iterative merging
                while ids.len() >= 2 {
                    let mut best_idx = None;
                    let mut best_merge_id = u32::MAX;

                    // Find earliest merge (lowest ID = earliest in training)
                    for i in 0..ids.len() - 1 {
                        let pair = unsafe { (*ids.get_unchecked(i), *ids.get_unchecked(i + 1)) };

                        if let Some(&merge_id) = self.merges.get(&pair) {
                            if merge_id < best_merge_id {
                                best_merge_id = merge_id;
                                best_idx = Some(i);
                            }
                        }
                    }

                    if let Some(idx) = best_idx {
                        ids[idx] = best_merge_id;
                        ids.remove(idx + 1);
                    } else {
                        break;
                    }
                }

                result.extend(ids);
            }
        }
        result
    }

    /// Batch encoding for better throughput
    pub fn encode_batch(&self, texts: Vec<&str>) -> Vec<Vec<u32>> {
        if texts.len() < 100 {
            // Sequential for small batches
            texts.into_iter().map(|t| self.encode(t)).collect()
        } else {
            // Parallel for large batches
            texts.par_iter().map(|t| self.encode(t)).collect()
        }
    }

    /// Export merges for serialization
    pub fn get_merges(&self) -> StdHashMap<(u32, u32), u32> {
        self.merges.clone()
    }

    /// Load pre-trained merges
    pub fn load_merges(&mut self, merges: StdHashMap<(u32, u32), u32>) {
        self.merges = merges;
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

// module registration
#[pymodule]
fn rust_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
