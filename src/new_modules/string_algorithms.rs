//! String Algorithms Module for NumRS2
//!
//! This module provides efficient implementations of fundamental string algorithms
//! suitable for bioinformatics, text mining, information retrieval, and data compression.
//!
//! # Overview
//!
//! The module is organized into five main areas:
//!
//! - **Pattern Matching**: KMP, Rabin-Karp, and Boyer-Moore-Horspool search
//! - **String Distances**: Levenshtein edit distance, longest common subsequence
//! - **Suffix Structures**: Suffix array with LCP array and binary search
//! - **Text Transforms**: Burrows-Wheeler Transform (encode/decode)
//! - **Hashing**: FNV-1a and djb2 non-cryptographic hash functions
//!
//! # Design Philosophy
//!
//! All algorithms are implemented natively in pure Rust following the approach of
//! `scirs2-core` 0.3.0's string algorithms infrastructure. No `unwrap()` calls
//! appear in production paths. All functions operate on byte slices (`&[u8]`) for
//! maximum flexibility, with `&str` convenience wrappers where appropriate.
//!
//! # Algorithmic References
//!
//! - Knuth, D. E., Morris, J. H., Pratt, V. R. (1977). Fast pattern matching in strings.
//! - Karp, R. M., Rabin, M. O. (1987). Efficient randomized pattern-matching algorithms.
//! - Boyer, R. S., Moore, J. S. (1977). A fast string searching algorithm.
//! - Horspool, R. N. (1980). Practical fast searching in strings.
//! - Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals.
//! - Burrows, M., Wheeler, D. J. (1994). A block-sorting lossless data compression algorithm.
//! - Kasai, T. et al. (2001). Linear-time longest-common-prefix computation in suffix arrays.

// ---------------------------------------------------------------------------
// Pattern Matching: Knuth-Morris-Pratt (KMP)
// ---------------------------------------------------------------------------

/// Build the KMP failure (partial match) table for `pattern`.
///
/// `failure[i]` is the length of the longest proper prefix of `pattern[0..=i]`
/// that is also a suffix. This table enables O(n + m) pattern matching.
fn kmp_failure_function(pattern: &[u8]) -> Vec<usize> {
    let m = pattern.len();
    if m == 0 {
        return Vec::new();
    }
    let mut failure = vec![0usize; m];
    let mut k = 0usize;
    let mut i = 1usize;
    while i < m {
        while k > 0 && pattern[k] != pattern[i] {
            k = failure[k - 1];
        }
        if pattern[k] == pattern[i] {
            k += 1;
        }
        failure[i] = k;
        i += 1;
    }
    failure
}

/// Knuth-Morris-Pratt pattern search.
///
/// Returns all (zero-based) starting positions of every (possibly overlapping)
/// occurrence of `pattern` in `text`. Runs in O(n + m) time where n = text length,
/// m = pattern length.
///
/// Returns an empty vector if `pattern` is empty or longer than `text`.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::kmp_search;
///
/// let positions = kmp_search(b"aababcab", b"ab");
/// assert_eq!(positions, vec![1, 3, 6]);
/// ```
pub fn kmp_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    let failure = kmp_failure_function(pattern);
    let mut results = Vec::new();
    let mut q = 0usize; // number of chars matched so far

    for (i, &c) in text.iter().enumerate() {
        while q > 0 && pattern[q] != c {
            q = failure[q - 1];
        }
        if pattern[q] == c {
            q += 1;
        }
        if q == m {
            results.push(i + 1 - m);
            q = failure[q - 1];
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Pattern Matching: Rabin-Karp (Rolling Hash)
// ---------------------------------------------------------------------------

/// Rabin-Karp single-pattern search using a polynomial rolling hash.
///
/// Returns all (possibly overlapping) starting positions of `pattern` in `text`.
/// Candidate positions identified by hash equality are verified with exact byte
/// comparison to eliminate false positives.
///
/// Uses base = 131, modulus = 1_000_000_007 for the rolling hash.
///
/// Returns an empty vector if `pattern` is empty or longer than `text`.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::rabin_karp_search;
///
/// let positions = rabin_karp_search(b"ababab", b"ab");
/// assert_eq!(positions, vec![0, 2, 4]);
/// ```
pub fn rabin_karp_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    const BASE: u64 = 131;
    const MOD: u64 = 1_000_000_007;

    // Compute base^(m-1) mod MOD for removing the leading character.
    let mut high_power = 1u64;
    for _ in 0..m.saturating_sub(1) {
        high_power = high_power.wrapping_mul(BASE) % MOD;
    }

    // Hash the pattern.
    let mut pat_hash = 0u64;
    for &b in pattern {
        pat_hash = (pat_hash.wrapping_mul(BASE) + b as u64) % MOD;
    }

    // Hash the first window of text.
    let mut txt_hash = 0u64;
    for &b in &text[..m] {
        txt_hash = (txt_hash.wrapping_mul(BASE) + b as u64) % MOD;
    }

    let mut results = Vec::new();

    // Check first window.
    if txt_hash == pat_hash && text[..m] == *pattern {
        results.push(0);
    }

    // Slide the window.
    for i in 1..=(n - m) {
        // Remove leading byte, add trailing byte.
        let remove_contribution = (text[i - 1] as u64).wrapping_mul(high_power) % MOD;
        let h = (txt_hash + MOD - remove_contribution) % MOD;
        txt_hash = (h.wrapping_mul(BASE) + text[i + m - 1] as u64) % MOD;

        if txt_hash == pat_hash && text[i..i + m] == *pattern {
            results.push(i);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Pattern Matching: Boyer-Moore-Horspool
// ---------------------------------------------------------------------------

/// Boyer-Moore-Horspool single-pattern search.
///
/// Returns all (zero-based) starting positions of every occurrence of `pattern`
/// in `text`. Uses a bad-character shift table for sub-linear average-case
/// performance.
///
/// This implementation finds all occurrences (including overlapping ones) by
/// advancing one position past each match.
///
/// Returns an empty vector if `pattern` is empty or longer than `text`.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::boyer_moore_search;
///
/// let positions = boyer_moore_search(b"AABAAABAAABAA", b"AAB");
/// assert_eq!(positions, vec![0, 4, 8]);
/// ```
pub fn boyer_moore_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    // Build the bad-character shift table.
    // Default shift is pattern length; for bytes in the pattern (excluding the last),
    // shift = distance from rightmost occurrence (excluding last) to end - 1.
    let mut shift = [m; 256];
    for (i, &b) in pattern[..m - 1].iter().enumerate() {
        shift[b as usize] = m - 1 - i;
    }

    let mut results = Vec::new();
    let mut i = 0usize; // starting index of current alignment

    while i + m <= n {
        // Compare from right to left.
        let mut j = m;
        while j > 0 && text[i + j - 1] == pattern[j - 1] {
            j -= 1;
        }
        if j == 0 {
            // Full match found.
            results.push(i);
            i += 1; // advance by 1 to find overlapping matches
        } else {
            // Shift by bad-character rule based on the last text character in the window.
            let bad_char_shift = shift[text[i + m - 1] as usize];
            i += if bad_char_shift == 0 {
                1
            } else {
                bad_char_shift
            };
        }
    }

    results
}

// ---------------------------------------------------------------------------
// String Distances: Levenshtein Edit Distance
// ---------------------------------------------------------------------------

/// Compute the Levenshtein edit distance between strings `a` and `b`.
///
/// The edit distance is the minimum number of single-character insertions,
/// deletions, or substitutions required to transform `a` into `b`.
///
/// Uses a space-optimized two-row dynamic programming approach in O(|a| * |b|)
/// time and O(min(|a|, |b|)) space.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("", "abc"), 3);
/// assert_eq!(levenshtein_distance("same", "same"), 0);
/// ```
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Two-row rolling DP.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// String Distances: Longest Common Subsequence
// ---------------------------------------------------------------------------

/// Compute the length of the longest common subsequence (LCS) of `a` and `b`.
///
/// Uses a space-optimized two-row DP approach in O(|a| * |b|) time.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::lcs_length;
///
/// assert_eq!(lcs_length("ABCBDAB", "BDCAB"), 4);
/// assert_eq!(lcs_length("abc", "abc"), 3);
/// assert_eq!(lcs_length("abc", "xyz"), 0);
/// ```
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return 0;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        for x in curr.iter_mut() {
            *x = 0;
        }
    }

    prev[n]
}

/// Reconstruct one actual longest common subsequence from `a` and `b`.
///
/// Builds the full O(|a| * |b|) DP table and then back-traces to recover the
/// actual subsequence string.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::lcs_string;
///
/// let result = lcs_string("ABCBDAB", "BDCAB");
/// assert_eq!(result.len(), 4);
/// ```
pub fn lcs_string(a: &str, b: &str) -> String {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();
    if m == 0 || n == 0 {
        return String::new();
    }

    // Full table for back-tracing.
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a_bytes[i - 1] == b_bytes[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Back-trace to recover the actual LCS.
    let mut result = Vec::with_capacity(dp[m][n]);
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if a_bytes[i - 1] == b_bytes[j - 1] {
            result.push(a_bytes[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();

    // Safe conversion: since both inputs are valid UTF-8 strings, and the LCS
    // consists of bytes selected from matching positions, the result bytes
    // form valid UTF-8 when both inputs use only single-byte (ASCII) characters.
    // For multi-byte UTF-8, the byte-level LCS may not be valid UTF-8, so we
    // use from_utf8_lossy for safety.
    String::from_utf8(result).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

// ---------------------------------------------------------------------------
// Suffix Structures: Suffix Array
// ---------------------------------------------------------------------------

/// Suffix array with LCP (Longest Common Prefix) array support.
///
/// Construction uses the O(n log n) prefix-doubling (Manber-Myers) algorithm.
/// The LCP array is built with Kasai's O(n) algorithm.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::SuffixArray;
///
/// let sa = SuffixArray::new(b"banana");
/// let positions = sa.search(b"ana");
/// assert_eq!(positions.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SuffixArray {
    /// The original text (owned copy).
    text: Vec<u8>,
    /// `sa[i]` = starting index of the i-th lexicographically smallest suffix.
    sa: Vec<usize>,
    /// `isa[j]` = rank (position in sorted order) of the suffix starting at j.
    isa: Vec<usize>,
    /// `lcp[i]` = length of the longest common prefix between `sa[i-1]` and `sa[i]`.
    /// `lcp[0]` is conventionally 0.
    lcp: Vec<usize>,
}

impl SuffixArray {
    /// Build a suffix array from a byte slice using O(n log n) prefix doubling.
    ///
    /// The text is stored internally so that `search` can be called without
    /// passing the text again.
    pub fn new(text: &[u8]) -> Self {
        let n = text.len();
        if n == 0 {
            return SuffixArray {
                text: Vec::new(),
                sa: Vec::new(),
                isa: Vec::new(),
                lcp: Vec::new(),
            };
        }

        // Initial rank based on the first byte of each suffix.
        let mut sa: Vec<usize> = (0..n).collect();
        let mut rank: Vec<i64> = text.iter().map(|&b| b as i64).collect();

        // Stable sort by initial rank.
        sa.sort_by_key(|&i| rank[i]);

        // Prefix-doubling: double the comparison key length each round.
        let mut gap = 1usize;
        while gap < n {
            let cur_rank = rank.clone();

            let second_rank = |i: usize| -> i64 {
                if i + gap < n {
                    cur_rank[i + gap]
                } else {
                    -1
                }
            };

            sa.sort_by(|&a, &b| {
                let ka = (cur_rank[a], second_rank(a));
                let kb = (cur_rank[b], second_rank(b));
                ka.cmp(&kb)
            });

            // Rebuild ranks from the sorted order.
            let mut new_rank = vec![0i64; n];
            new_rank[sa[0]] = 0;
            for i in 1..n {
                let prev = sa[i - 1];
                let cur = sa[i];
                let same = cur_rank[prev] == cur_rank[cur] && second_rank(prev) == second_rank(cur);
                new_rank[cur] = new_rank[prev] + if same { 0 } else { 1 };
            }
            rank = new_rank;

            // Early exit when all ranks are unique.
            if rank[sa[n - 1]] == (n as i64 - 1) {
                break;
            }

            gap <<= 1;
        }

        // Build inverse suffix array.
        let mut isa = vec![0usize; n];
        for (i, &s) in sa.iter().enumerate() {
            isa[s] = i;
        }

        // Build LCP array with Kasai's algorithm.
        let lcp = Self::build_lcp(text, &sa, &isa);

        SuffixArray {
            text: text.to_vec(),
            sa,
            isa,
            lcp,
        }
    }

    /// Kasai's O(n) LCP array construction.
    fn build_lcp(text: &[u8], sa: &[usize], isa: &[usize]) -> Vec<usize> {
        let n = text.len();
        let mut lcp = vec![0usize; n];
        let mut h = 0usize;
        for i in 0..n {
            if isa[i] > 0 {
                let j = sa[isa[i] - 1];
                while i + h < n && j + h < n && text[i + h] == text[j + h] {
                    h += 1;
                }
                lcp[isa[i]] = h;
                h = h.saturating_sub(1);
            }
        }
        lcp
    }

    /// Search for all occurrences of `pattern` in the text.
    ///
    /// Returns a sorted vector of starting positions. Uses binary search
    /// on the suffix array for O(m log n) search time.
    pub fn search(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() || self.text.is_empty() {
            return Vec::new();
        }
        let n = self.text.len();
        let m = pattern.len();

        // Lower bound: first suffix >= pattern.
        let lo = {
            let mut low = 0usize;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                let start = self.sa[mid];
                let end = (start + m).min(n);
                if self.text[start..end] < *pattern {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };

        // Upper bound: first suffix > pattern.
        let hi = {
            let mut low = lo;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                let start = self.sa[mid];
                let end = (start + m).min(n);
                if self.text[start..end] <= *pattern {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };

        if lo < hi {
            let mut positions: Vec<usize> = self.sa[lo..hi].to_vec();
            positions.sort_unstable();
            positions
        } else {
            Vec::new()
        }
    }

    /// Return the LCP (Longest Common Prefix) array.
    ///
    /// `lcp_array()[i]` is the length of the longest common prefix between
    /// the (i-1)-th and i-th lexicographically sorted suffixes. `lcp_array()[0]`
    /// is conventionally 0.
    pub fn lcp_array(&self) -> Vec<usize> {
        self.lcp.clone()
    }

    /// Return a reference to the underlying suffix array.
    pub fn sa(&self) -> &[usize] {
        &self.sa
    }

    /// Return a reference to the inverse suffix array.
    pub fn isa(&self) -> &[usize] {
        &self.isa
    }

    /// Return a reference to the stored text.
    pub fn text(&self) -> &[u8] {
        &self.text
    }
}

// ---------------------------------------------------------------------------
// Text Transforms: Burrows-Wheeler Transform
// ---------------------------------------------------------------------------

/// Encode `text` with the Burrows-Wheeler Transform.
///
/// Returns `(bwt_bytes, original_row)` where `original_row` is the row in the
/// sorted rotation matrix corresponding to the original string, needed for decoding.
///
/// The BWT is derived efficiently from the suffix array without constructing all
/// rotations explicitly.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::{bwt_encode, bwt_decode};
///
/// let (bwt, pos) = bwt_encode(b"banana");
/// let decoded = bwt_decode(&bwt, pos);
/// assert_eq!(decoded, b"banana");
/// ```
pub fn bwt_encode(text: &[u8]) -> (Vec<u8>, usize) {
    let n = text.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Build suffix array and derive the BWT directly.
    let sa = SuffixArray::new(text);
    let mut bwt = Vec::with_capacity(n);
    let mut original_pos = 0usize;

    for (i, &start) in sa.sa.iter().enumerate() {
        if start == 0 {
            bwt.push(text[n - 1]);
            original_pos = i;
        } else {
            bwt.push(text[start - 1]);
        }
    }

    (bwt, original_pos)
}

/// Decode a Burrows-Wheeler Transform.
///
/// `bwt` is the transformed byte sequence and `idx` is the original row index
/// returned by [`bwt_encode`].
///
/// Reconstruction uses the LF-mapping (Last-to-First column mapping) which
/// runs in O(n) time after an O(n) preprocessing step.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::{bwt_encode, bwt_decode};
///
/// let (bwt, pos) = bwt_encode(b"mississippi");
/// let decoded = bwt_decode(&bwt, pos);
/// assert_eq!(decoded, b"mississippi");
/// ```
pub fn bwt_decode(bwt: &[u8], idx: usize) -> Vec<u8> {
    let n = bwt.len();
    if n == 0 {
        return Vec::new();
    }

    // Count occurrences of each byte.
    let mut count = [0usize; 256];
    for &b in bwt {
        count[b as usize] += 1;
    }

    // Compute starting position of each byte in the first column (sorted BWT).
    let mut first_occ = [0usize; 256];
    let mut total = 0usize;
    for (b, c) in count.iter().enumerate() {
        first_occ[b] = total;
        total += c;
    }

    // Build the LF-mapping.
    let mut byte_rank = [0usize; 256];
    let mut lf = vec![0usize; n];
    for (i, &b) in bwt.iter().enumerate() {
        lf[i] = first_occ[b as usize] + byte_rank[b as usize];
        byte_rank[b as usize] += 1;
    }

    // Reconstruct original string by following LF-mapping backwards.
    let mut result = vec![0u8; n];
    let mut row = idx;
    for slot in result.iter_mut().rev() {
        *slot = bwt[row];
        row = lf[row];
    }

    result
}

// ---------------------------------------------------------------------------
// Hashing: FNV-1a
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash of a byte slice.
///
/// A fast, non-cryptographic hash function suitable for hash-map keys
/// and general fingerprinting. Produces well-distributed hashes for
/// short strings.
///
/// The FNV-1a variant XORs each byte into the hash before multiplying,
/// providing better avalanche characteristics than FNV-1.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::fnv1a_hash;
///
/// let h = fnv1a_hash(b"hello");
/// assert_ne!(h, 0);
/// // FNV-1a of empty data is the offset basis.
/// assert_eq!(fnv1a_hash(b""), 14_695_981_039_346_656_037u64);
/// ```
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Hashing: djb2
// ---------------------------------------------------------------------------

/// djb2 64-bit hash of a byte slice.
///
/// Classic algorithm attributed to Dan Bernstein. Uses the formula
/// `hash = hash * 33 + byte` starting from the magic constant 5381.
///
/// Simple, fast, and produces reasonable distribution for ASCII text.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::string_algorithms::djb2_hash;
///
/// let h = djb2_hash(b"hello");
/// assert_ne!(h, 0);
/// ```
pub fn djb2_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    for &b in data {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- KMP ----------------------------------------------------------------

    #[test]
    fn test_string_algo_kmp_basic() {
        let pos = kmp_search(b"aababcab", b"ab");
        assert_eq!(pos, vec![1, 3, 6]);
    }

    #[test]
    fn test_string_algo_kmp_no_match() {
        let pos = kmp_search(b"hello", b"xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_string_algo_kmp_overlapping() {
        let pos = kmp_search(b"aaa", b"aa");
        assert_eq!(pos, vec![0, 1]);
    }

    #[test]
    fn test_string_algo_kmp_full_text_match() {
        let pos = kmp_search(b"abcabc", b"abcabc");
        assert_eq!(pos, vec![0]);
    }

    // ---- Rabin-Karp ---------------------------------------------------------

    #[test]
    fn test_string_algo_rabin_karp_basic() {
        let pos = rabin_karp_search(b"ababab", b"ab");
        assert_eq!(pos, vec![0, 2, 4]);
    }

    #[test]
    fn test_string_algo_rabin_karp_no_match() {
        let pos = rabin_karp_search(b"hello world", b"xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_string_algo_rabin_karp_overlapping() {
        let pos = rabin_karp_search(b"aaa", b"aa");
        assert_eq!(pos, vec![0, 1]);
    }

    // ---- Boyer-Moore-Horspool -----------------------------------------------

    #[test]
    fn test_string_algo_boyer_moore_basic() {
        let pos = boyer_moore_search(b"AABAAABAAABAA", b"AAB");
        assert_eq!(pos, vec![0, 4, 8]);
    }

    #[test]
    fn test_string_algo_boyer_moore_no_match() {
        let pos = boyer_moore_search(b"hello world", b"xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_string_algo_boyer_moore_single_char() {
        let pos = boyer_moore_search(b"aaa", b"a");
        assert_eq!(pos, vec![0, 1, 2]);
    }

    // ---- Levenshtein --------------------------------------------------------

    #[test]
    fn test_string_algo_levenshtein_kitten_sitting() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_string_algo_levenshtein_empty() {
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn test_string_algo_levenshtein_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_string_algo_levenshtein_single_substitution() {
        assert_eq!(levenshtein_distance("abc", "axc"), 1);
    }

    // ---- LCS ----------------------------------------------------------------

    #[test]
    fn test_string_algo_lcs_length_basic() {
        assert_eq!(lcs_length("ABCBDAB", "BDCAB"), 4);
    }

    #[test]
    fn test_string_algo_lcs_length_identical() {
        assert_eq!(lcs_length("abc", "abc"), 3);
    }

    #[test]
    fn test_string_algo_lcs_length_disjoint() {
        assert_eq!(lcs_length("abc", "xyz"), 0);
    }

    #[test]
    fn test_string_algo_lcs_string_length() {
        let result = lcs_string("ABCBDAB", "BDCAB");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_string_algo_lcs_string_valid_subsequence() {
        let a = "ABCBDAB";
        let b = "BDCAB";
        let seq = lcs_string(a, b);
        // Verify seq is a subsequence of a.
        let a_bytes = a.as_bytes();
        let mut ai = 0usize;
        for c in seq.bytes() {
            while ai < a_bytes.len() && a_bytes[ai] != c {
                ai += 1;
            }
            assert!(ai < a_bytes.len(), "LCS element not found in a");
            ai += 1;
        }
    }

    // ---- Suffix Array -------------------------------------------------------

    #[test]
    fn test_string_algo_suffix_array_banana() {
        let sa = SuffixArray::new(b"banana");
        // Sorted suffixes: a, ana, anana, banana, na, nana
        // Starting positions: 5, 3, 1, 0, 4, 2
        assert_eq!(sa.sa(), &[5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_string_algo_suffix_array_search() {
        let sa = SuffixArray::new(b"banana");
        let mut positions = sa.search(b"ana");
        positions.sort_unstable();
        assert_eq!(positions, vec![1, 3]);
    }

    #[test]
    fn test_string_algo_suffix_array_search_not_found() {
        let sa = SuffixArray::new(b"banana");
        let positions = sa.search(b"xyz");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_string_algo_suffix_array_lcp() {
        let sa = SuffixArray::new(b"banana");
        let lcp = sa.lcp_array();
        assert_eq!(lcp[0], 0);
        assert_eq!(lcp[1], 1); // "a" vs "ana"
        assert_eq!(lcp[2], 3); // "ana" vs "anana"
        assert_eq!(lcp[3], 0); // "anana" vs "banana"
        assert_eq!(lcp[4], 0); // "banana" vs "na"
        assert_eq!(lcp[5], 2); // "na" vs "nana"
    }

    #[test]
    fn test_string_algo_suffix_array_empty() {
        let sa = SuffixArray::new(b"");
        assert!(sa.sa().is_empty());
        assert!(sa.lcp_array().is_empty());
    }

    // ---- BWT ----------------------------------------------------------------

    #[test]
    fn test_string_algo_bwt_roundtrip_banana() {
        let (bwt, pos) = bwt_encode(b"banana");
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, b"banana");
    }

    #[test]
    fn test_string_algo_bwt_roundtrip_mississippi() {
        let (bwt, pos) = bwt_encode(b"mississippi");
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, b"mississippi");
    }

    #[test]
    fn test_string_algo_bwt_roundtrip_empty() {
        let (bwt, pos) = bwt_encode(b"");
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, b"");
    }

    #[test]
    fn test_string_algo_bwt_roundtrip_repeated() {
        let original = b"aaaaaa";
        let (bwt, pos) = bwt_encode(original);
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, original.as_ref());
    }

    // ---- Hashing ------------------------------------------------------------

    #[test]
    fn test_string_algo_fnv1a_deterministic() {
        assert_eq!(fnv1a_hash(b"hello"), fnv1a_hash(b"hello"));
    }

    #[test]
    fn test_string_algo_fnv1a_different_strings() {
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    #[test]
    fn test_string_algo_fnv1a_empty() {
        assert_eq!(fnv1a_hash(b""), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_string_algo_djb2_deterministic() {
        assert_eq!(djb2_hash(b"hello"), djb2_hash(b"hello"));
    }

    #[test]
    fn test_string_algo_djb2_different_strings() {
        assert_ne!(djb2_hash(b"foo"), djb2_hash(b"bar"));
    }

    // ---- Cross-algorithm consistency ----------------------------------------

    #[test]
    fn test_string_algo_all_search_agree() {
        // All three pattern matching algorithms should find the same positions.
        let text = b"abcabcabc";
        let pattern = b"abc";
        let kmp_result = kmp_search(text, pattern);
        let rk_result = rabin_karp_search(text, pattern);
        let bm_result = boyer_moore_search(text, pattern);
        assert_eq!(kmp_result, rk_result);
        assert_eq!(kmp_result, bm_result);
    }

    #[test]
    fn test_string_algo_pattern_longer_than_text() {
        // All algorithms should return empty for pattern longer than text.
        assert!(kmp_search(b"ab", b"abc").is_empty());
        assert!(rabin_karp_search(b"ab", b"abc").is_empty());
        assert!(boyer_moore_search(b"ab", b"abc").is_empty());
    }
}
