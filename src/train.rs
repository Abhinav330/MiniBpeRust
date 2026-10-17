use fancy_regex::Regex;
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::fs;

pub type Pair = (u32, u32);

// ---------------------------------------------------------------------------
// Step 1: split text into chunks and count every unique chunk once.
// Chunks are remembered in order of first appearance (needed to match
// minbpe's tie-breaking exactly).

pub struct ChunkCounter {
    map: FxHashMap<Vec<u8>, (usize, u64)>, // chunk bytes -> (first-seen order, count)
    next: usize,
}

impl ChunkCounter {
    pub fn new() -> Self {
        Self { map: FxHashMap::default(), next: 0 }
    }

    pub fn unique(&self) -> usize {
        self.map.len()
    }

    pub fn add_text(&mut self, re: &Regex, text: &str) -> Result<(), String> {
        for m in re.find_iter(text) {
            let m = m.map_err(|e| e.to_string())?;
            let bytes = m.as_str().as_bytes();
            match self.map.get_mut(bytes) {
                Some(entry) => entry.1 += 1,
                None => {
                    self.map.insert(bytes.to_vec(), (self.next, 1));
                    self.next += 1;
                }
            }
        }
        Ok(())
    }

    /// Returns unique chunks (as bytes) in first-seen order, with their counts.
    pub fn finish(self) -> (Vec<Vec<u8>>, Vec<u64>) {
        let mut items: Vec<(Vec<u8>, usize, u64)> = self
            .map
            .into_iter()
            .map(|(word, (order, count))| (word, order, count))
            .collect();
        items.sort_unstable_by_key(|t| t.1);
        let mut words = Vec::with_capacity(items.len());
        let mut counts = Vec::with_capacity(items.len());
        for (word, _, count) in items {
            words.push(word);
            counts.push(count);
        }
        (words, counts)
    }
}

// ---------------------------------------------------------------------------
// Step 2: the BPE training loop, working on the unique-chunk table.
//
// minbpe picks max(stats, key=stats.get): the highest count, and on a tie the
// pair that appears EARLIEST in the corpus. We reproduce that: among tied
// pairs, the winner is the one whose first occurrence has the smallest
// (chunk order, position in chunk).

fn first_occurrence(
    p: Pair,
    list: &[u32],
    ids: &[u32],
    start: &[usize],
    wlen: &[u32],
) -> (usize, usize) {
    let mut best = (usize::MAX, usize::MAX);
    for &w in list {
        let w = w as usize;
        if w > best.0 {
            continue;
        }
        let s = start[w];
        let l = wlen[w] as usize;
        let word = &ids[s..s + l];
        for j in 0..l.saturating_sub(1) {
            if word[j] == p.0 && word[j + 1] == p.1 {
                if (w, j) < best {
                    best = (w, j);
                }
                break;
            }
        }
    }
    best
}

/// Returns the merges in order, each with the count it had when chosen.
pub fn train_core(
    words: Vec<Vec<u8>>,
    counts: Vec<u64>,
    num_merges: u32,
    verbose: bool,
) -> Vec<(Pair, u64)> {
    let n_words = words.len();

    // All chunks stored in one flat array of token ids (compact in memory).
    let total: usize = words.iter().map(|w| w.len()).sum();
    let mut ids: Vec<u32> = Vec::with_capacity(total);
    let mut start: Vec<usize> = Vec::with_capacity(n_words);
    let mut wlen: Vec<u32> = Vec::with_capacity(n_words);
    for w in words {
        start.push(ids.len());
        wlen.push(w.len() as u32);
        ids.extend(w.iter().map(|&b| b as u32));
    }

    // pair -> total count, and pair -> list of chunks that (may) contain it
    let mut pair_count: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut pair_words: FxHashMap<Pair, Vec<u32>> = FxHashMap::default();
    for w in 0..n_words {
        let s = start[w];
        let l = wlen[w] as usize;
        if l < 2 {
            continue;
        }
        let c = counts[w] as i64;
        for j in 0..l - 1 {
            let p = (ids[s + j], ids[s + j + 1]);
            *pair_count.entry(p).or_insert(0) += c;
            let list = pair_words.entry(p).or_default();
            if list.last() != Some(&(w as u32)) {
                list.push(w as u32);
            }
        }
    }

    // Pairs sorted by count, so the best pair (and all ties) are easy to find.
    let mut by_count: BTreeSet<(i64, Pair)> =
        pair_count.iter().map(|(&p, &c)| (c, p)).collect();

    let mut merges: Vec<(Pair, u64)> = Vec::with_capacity(num_merges as usize);
    let mut delta: FxHashMap<Pair, i64> = FxHashMap::default();

    for i in 0..num_merges {
        // 1. highest count
        let top_count = match by_count.iter().next_back() {
            Some(&(c, _)) => c,
            None => break, // nothing left to merge
        };

        // 2. all pairs tied at that count
        let candidates: Vec<Pair> = by_count
            .range((top_count, (0u32, 0u32))..)
            .map(|&(_, p)| p)
            .collect();

        // 3. break ties like minbpe: earliest first occurrence wins
        let best: Pair = if candidates.len() == 1 {
            candidates[0]
        } else {
            let mut best_pair = candidates[0];
            let mut best_key = (usize::MAX, usize::MAX);
            for &p in &candidates {
                let list: &[u32] = pair_words
                    .get(&p)
                    .expect("candidate pair must have a chunk list")
                    .as_slice();
                let key = first_occurrence(p, list, &ids, &start, &wlen);
                if key < best_key {
                    best_key = key;
                    best_pair = p;
                }
            }
            best_pair
        };

        let new_id: u32 = 256 + i;

        // 4. apply the merge to every chunk that contains the pair
        let mut affected = pair_words
            .remove(&best)
            .expect("best pair must have a chunk list");
        affected.sort_unstable();
        affected.dedup();

        for &w in &affected {
            let w = w as usize;
            let s = start[w];
            let l = wlen[w] as usize;
            let c = counts[w] as i64;

            let mut contains = false;
            for j in 0..l.saturating_sub(1) {
                if ids[s + j] == best.0 && ids[s + j + 1] == best.1 {
                    contains = true;
                    break;
                }
            }
            if !contains {
                continue; // stale entry
            }

            // remove this chunk's old pairs from the counts
            for j in 0..l - 1 {
                *delta.entry((ids[s + j], ids[s + j + 1])).or_insert(0) -= c;
            }

            // merge left to right, no overlap (same as minbpe's merge())
            let mut read = 0usize;
            let mut write = 0usize;
            while read < l {
                if read + 1 < l && ids[s + read] == best.0 && ids[s + read + 1] == best.1 {
                    ids[s + write] = new_id;
                    read += 2;
                } else {
                    ids[s + write] = ids[s + read];
                    read += 1;
                }
                write += 1;
            }
            wlen[w] = write as u32;

            // add this chunk's new pairs
            for j in 0..write.saturating_sub(1) {
                let p = (ids[s + j], ids[s + j + 1]);
                *delta.entry(p).or_insert(0) += c;
                if p.0 == new_id || p.1 == new_id {
                    let list = pair_words.entry(p).or_default();
                    if list.last() != Some(&(w as u32)) {
                        list.push(w as u32);
                    }
                }
            }
        }

        // 5. apply all count changes at once
        for (p, d) in delta.drain() {
            if d == 0 {
                continue;
            }
            let old = pair_count.get(&p).copied().unwrap_or(0);
            let new = old + d;
            if old > 0 {
                by_count.remove(&(old, p));
            }
            if new > 0 {
                by_count.insert((new, p));
                pair_count.insert(p, new);
            } else {
                pair_count.remove(&p);
            }
        }

        merges.push((best, top_count as u64));
        if verbose {
            println!(
                "merge {}/{}: {:?} -> {} had {} occurrences",
                i + 1,
                num_merges,
                best,
                new_id,
                top_count
            );
        }
    }

    merges
}

// ---------------------------------------------------------------------------
// Public entry points used by lib.rs

fn check_vocab(vocab_size: u32) -> Result<(), String> {
    if vocab_size < 256 {
        return Err("vocab_size must be >= 256".to_string());
    }
    Ok(())
}

fn finish_and_train(
    counter: ChunkCounter,
    vocab_size: u32,
    verbose: bool,
) -> Result<Vec<(Pair, u64)>, String> {
    let (words, counts) = counter.finish();
    if verbose {
        println!("{} unique chunks, starting merges", words.len());
    }
    Ok(train_core(words, counts, vocab_size - 256, verbose))
}

pub fn train_on_text(
    text: &str,
    pattern: &str,
    vocab_size: u32,
    verbose: bool,
) -> Result<Vec<(Pair, u64)>, String> {
    check_vocab(vocab_size)?;
    let re = Regex::new(pattern).map_err(|e| e.to_string())?;
    let mut counter = ChunkCounter::new();
    counter.add_text(&re, text)?;
    finish_and_train(counter, vocab_size, verbose)
}

pub fn train_on_files(
    paths: &[String],
    pattern: &str,
    vocab_size: u32,
    verbose: bool,
) -> Result<Vec<(Pair, u64)>, String> {
    check_vocab(vocab_size)?;
    let re = Regex::new(pattern).map_err(|e| e.to_string())?;
    let mut counter = ChunkCounter::new();
    for path in paths {
        let bytes = fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        let text = String::from_utf8(bytes)
            .map_err(|e| format!("{}: input is not valid UTF-8: {}", path, e))?;
        counter.add_text(&re, &text)?;
        if verbose {
            println!("counted {} -> {} unique chunks so far", path, counter.unique());
        }
    }
    finish_and_train(counter, vocab_size, verbose)
}