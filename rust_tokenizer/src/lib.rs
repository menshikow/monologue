use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;
use std::sync::Mutex;
// better than the standard binary heap: has 8 children per node instead of 2
// which fits better in the CPU cache lines, reducing RAM lookups
use ahash::{AHashMap, AHashSet}; // faster hashing algorithm than the secure std default
use compact_str::CompactString; // more eficient strings
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*; // turn standard interators into parallel ones

const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
type Pair = (u32, u32);

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(6);
        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
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
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Regex,
    pub special_tokens: StdHashMap<String, u32>,
}

impl Tokenizer {
    fn count_pairs_parallel(
        words: &[Word],
        counts: &[i32],
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .par_iter()
            .enumerate()
            .map(|(i, w)| {
                let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
                let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                if w.ids.len() >= 2 && counts[i] != 0 {
                    for (a, b) in w.pairs() {
                        *local_pc.entry((a, b)).or_default() += counts[i];
                        local_wtu.entry((a, b)).or_default().insert(i);
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

        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }

        let mut merges_done = 0;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else {
                break;
            };

            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 {
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }

            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

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

            for (pair, pos) in local_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }

            merges_done += 1;
        }
    }
}
