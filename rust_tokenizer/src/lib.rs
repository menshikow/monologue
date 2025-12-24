use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;
use std::sync::Arc;
// better than the standard binary heap: has 8 children per node instead of 2
// which fits better in the CPU cache lines, reducing RAM lookups
use ahash::{AHashMap, AHashSet}; // faster hashing algorithm than the secure std default
use compact_str::CompactString; // more efficient strings
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*; // turn standard iterators into parallel ones

const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
type Pair = (u32, u32);

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    // optimized merge with pre-allocated capacity and reduced allocations
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        // pre-allocate with exact or near-exact capacity
        let mut out = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(6);
        let mut i = 0;

        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                // record changes for pair count updates
                if let Some(&prev) = out.last() {
                    deltas.push(((prev, a), -1));
                    deltas.push(((prev, new_id), 1));
                }

                deltas.push(((a, b), -1));

                if i + 2 < n {
                    let next = self.ids[i + 2];
                    deltas.push(((b, next), -1));
                    deltas.push(((new_id, next), 1));
                }

                out.push(new_id);
                i += 2;
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // primary: count (descending), secondary: pair (for determinism)
        match self.count.cmp(&other.count) {
            Ordering::Equal => other.pair.cmp(&self.pair),
            other => other,
        }
    }
}

#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Arc<Regex>, // arc to enable cheap cloning
    pub special_tokens: StdHashMap<String, u32>,
}

impl Tokenizer {
    // optimized parallel pair counting with better chunking strategy
    fn count_pairs_parallel(
        words: &[Word],
        counts: &[i32],
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        // use chunk size that balances parallelism overhead
        let chunk_size = (words.len() / rayon::current_num_threads()).max(100);

        words
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;
                let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
                let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                for (offset, w) in chunk.iter().enumerate() {
                    let i = base_idx + offset;
                    if w.ids.len() >= 2 && counts[i] != 0 {
                        for pair in w.pairs() {
                            *local_pc.entry(pair).or_default() += counts[i];
                            local_wtu.entry(pair).or_default().insert(i);
                        }
                    }
                }
                (local_pc, local_wtu)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                    for (k, v) in pc {
                        *acc_pc.entry(k).or_default() += v;
                    }
                    for (k, s) in wtu {
                        acc_wtu.entry(k).or_default().extend(s);
                    }
                    (acc_pc, acc_wtu)
                },
            )
    }

    fn train_core(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size >= 256, "vocab_size must be >= 256");
        let num_merges = vocab_size - 256;

        let (mut pair_counts, mut where_to_update) = Self::count_pairs_parallel(&words, &counts);

        // pre-allocate heap with known capacity
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

        let mut merges_done = 0;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else {
                break;
            };

            // lazy evaluation: check if count is still valid
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 {
                if current > 0 {
                    top.count = current as u64;
                    heap.push(top);
                }
                continue;
            }

            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            // reuse allocation across iterations
            let mut local_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

            for &word_idx in &top.pos {
                let changes = words[word_idx].merge_pair(top.pair, new_id);

                for (pair, delta) in changes {
                    let total_change = delta * counts[word_idx];
                    *pair_counts.entry(pair).or_default() += total_change;

                    if delta > 0 {
                        local_updates.entry(pair).or_default().insert(word_idx);
                    }
                }
            }

            // push updated pairs back to heap
            for (pair, pos) in local_updates {
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
            merges: StdHashMap::new(),
            pattern: GPT4_PATTERN.to_string(),
            compiled_pattern: Arc::new(compiled_pattern),
            special_tokens: StdHashMap::new(),
        })
    }

    pub fn register_special_token(&mut self, token: String, id: u32) {
        self.special_tokens.insert(token, id);
    }

    // main training entry point for python: streaming data, parallel counting, and BPE training
    #[pyo3(signature = (iterator, vocab_size, buffer_size=10_000))]
    pub fn train_from_iterator(
        &mut self,
        py: Python<'_>,
        iterator: &Bound<'_, PyAny>,
        vocab_size: u32,
        buffer_size: usize,
    ) -> PyResult<()> {
        let py_iter = iterator.iter()?;

        // with_capacity for better allocation strategy
        let mut global_counts: AHashMap<CompactString, i32> = AHashMap::with_capacity(100_000);
        let mut buffer: Vec<String> = Vec::with_capacity(buffer_size);

        // clone arc for use in parallel context
        let pattern = Arc::clone(&self.compiled_pattern);

        loop {
            buffer.clear();

            // fill buffer
            for _ in 0..buffer_size {
                match py_iter.next() {
                    Some(Ok(item)) => {
                        buffer.push(item.extract()?);
                    }
                    Some(Err(e)) => return Err(e),
                    None => break,
                }
            }

            if buffer.is_empty() {
                break;
            }

            // process batch in parallel
            let batch_counts = py.allow_threads(|| {
                buffer
                    .par_iter()
                    .map(|text| {
                        let mut local_map = AHashMap::with_capacity(64);
                        for m in pattern.find_iter(text) {
                            if let Ok(m) = m {
                                *local_map
                                    .entry(CompactString::from(m.as_str()))
                                    .or_default() += 1;
                            }
                        }
                        local_map
                    })
                    .reduce(
                        || AHashMap::with_capacity(1024),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            // merge into global counts
            for (k, v) in batch_counts {
                *global_counts.entry(k).or_default() += v;
            }

            // check if buffer wasn't fully filled
            if buffer.len() < buffer_size {
                break;
            }
        }

        // convert to words and counts vectors
        let mut words = Vec::with_capacity(global_counts.len());
        let mut counts_vec = Vec::with_capacity(global_counts.len());

        for (token_str, count) in global_counts {
            let ids: Vec<u32> = token_str.bytes().map(|b| b as u32).collect();
            words.push(Word::new(ids));
            counts_vec.push(count);
        }

        self.train_core(words, counts_vec, vocab_size);
        Ok(())
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
