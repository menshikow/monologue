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
