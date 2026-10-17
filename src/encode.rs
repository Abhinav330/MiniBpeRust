use crate::train::Pair;
use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Fast replacement for RegexTokenizer.encode_ordinary / _encode_chunk.
/// Same logic as minbpe: repeatedly merge the pair with the lowest merge id.
pub struct Encoder {
    re: Regex,
    ranks: FxHashMap<Pair, u32>, // (id, id) -> new token id (lower = earlier merge)
}

impl Encoder {
    pub fn new(pattern: &str, merges: &[Pair]) -> Result<Self, String> {
        let re = Regex::new(pattern).map_err(|e| e.to_string())?;
        let mut ranks: FxHashMap<Pair, u32> = FxHashMap::default();
        for (i, &p) in merges.iter().enumerate() {
            ranks.insert(p, 256 + i as u32);
        }
        Ok(Self { re, ranks })
    }

    /// Encode one regex chunk (as bytes) into token ids.
    fn encode_chunk(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        while ids.len() >= 2 {
            // find the pair with the lowest merge id
            let mut best: Option<(u32, Pair)> = None;
            for w in ids.windows(2) {
                let p = (w[0], w[1]);
                if let Some(&id) = self.ranks.get(&p) {
                    if best.map_or(true, |(b, _)| id < b) {
                        best = Some((id, p));
                    }
                }
            }
            let (new_id, pair) = match best {
                Some(x) => x,
                None => break, // nothing left to merge
            };

            // merge all occurrences, left to right, no overlap (same as minbpe)
            let mut read = 0usize;
            let mut write = 0usize;
            while read < ids.len() {
                if read + 1 < ids.len() && ids[read] == pair.0 && ids[read + 1] == pair.1 {
                    ids[write] = new_id;
                    read += 2;
                } else {
                    ids[write] = ids[read];
                    read += 1;
                }
                write += 1;
            }
            ids.truncate(write);
        }
        ids
    }

    /// Number of merges this encoder knows about.
    pub fn num_merges(&self) -> usize {
        self.ranks.len()
    }

    /// Encode text, ignoring special tokens (Python layer handles those).
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut out: Vec<u32> = Vec::with_capacity(text.len() / 3 + 1);
        // repeated words are common, so remember chunks we have already encoded
        let mut cache: FxHashMap<&str, Vec<u32>> = FxHashMap::default();

        for m in self.re.find_iter(text) {
            let m = m.map_err(|e| e.to_string())?;
            let s = m.as_str();
            if let Some(ids) = cache.get(s) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.encode_chunk(s.as_bytes());
            out.extend_from_slice(&ids);
            if cache.len() < 1_000_000 {
                cache.insert(s, ids);
            }
        }
        Ok(out)
    }

    /// Encode many texts in parallel, one result list per text.
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>, String> {
        texts
            .par_iter()
            .map(|t| self.encode_ordinary(t))
            .collect::<Result<Vec<_>, _>>()
    }
}