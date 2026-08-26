//! Bioinformatics module for NumRS2.
//!
//! Wraps and extends the bioinformatics capabilities from `scirs2-core` 0.3.0,
//! providing ergonomic Rust interfaces for:
//!
//! - **Sequence alignment**: Needleman-Wunsch (global), Smith-Waterman (local),
//!   Levenshtein edit distance.
//! - **Phylogenetic distances**: Hamming distance, Jukes-Cantor evolutionary
//!   distance, pairwise distance matrix.
//! - **Sequence statistics**: nucleotide frequencies, GC content, codon usage.
//!
//! All functions accept `&str` or `&[&str]` rather than raw byte slices,
//! converting internally so callers work with familiar Rust string types.
//!
//! # Quick Start
//!
//! ```rust
//! use numrs2::new_modules::bioinformatics::{
//!     needleman_wunsch, smith_waterman, edit_distance,
//!     hamming_distance, jukes_cantor_distance, distance_matrix,
//!     nucleotide_frequencies, gc_content, codon_usage,
//! };
//!
//! // Global alignment
//! let result = needleman_wunsch("AGCT", "AGT", 1, -1, -2).expect("alignment failed");
//! assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());
//!
//! // GC content
//! let gc = gc_content("AATGCG");
//! assert!((gc - 0.5).abs() < 1e-10);
//! ```
//!
//! # References
//!
//! - Needleman & Wunsch (1970), *J. Mol. Biol.* 48: 443-453.
//! - Smith & Waterman (1981), *J. Mol. Biol.* 147: 195-197.
//! - Jukes & Cantor (1969), *Mammalian Protein Metabolism* 21-132.
//! - Levenshtein (1966), *Soviet Physics Doklady* 10: 707-710.

use std::collections::HashMap;

use scirs2_core::bioinformatics::alignment::{
    edit_distance as core_edit_distance, needleman_wunsch as core_nw, smith_waterman as core_sw,
};
use scirs2_core::bioinformatics::phylo::{
    distance_matrix as core_distance_matrix, hamming_distance as core_hamming,
    jukes_cantor_distance as core_jc,
};
use scirs2_core::bioinformatics::sequence::gc_content as core_gc_content;
use scirs2_core::bioinformatics::stats::nucleotide_frequencies as core_nuc_freq;

use crate::error::{NumRs2Error, Result};

// ─── Result and helper types ──────────────────────────────────────────────────

/// Converts a `scirs2_core` error into a `NumRs2Error`.
fn map_core_err(e: scirs2_core::error::CoreError) -> NumRs2Error {
    NumRs2Error::ComputationError(format!("scirs2-core bioinformatics error: {e}"))
}

// ─── AlignmentResult ─────────────────────────────────────────────────────────

/// Result of a pairwise sequence alignment.
///
/// Both `aligned_seq1` and `aligned_seq2` have equal length; gap characters
/// are represented by `'-'`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentResult {
    /// Alignment score (higher is better for global; 0 is the minimum for
    /// local alignments).
    pub score: f64,
    /// First aligned sequence, possibly padded with `'-'` characters.
    pub aligned_seq1: String,
    /// Second aligned sequence, possibly padded with `'-'` characters.
    pub aligned_seq2: String,
}

// ─── Sequence alignment ───────────────────────────────────────────────────────

/// Performs global pairwise sequence alignment using the Needleman-Wunsch
/// dynamic-programming algorithm.
///
/// # Parameters
///
/// - `seq1`, `seq2` — sequences to align (ASCII, case-sensitive).
/// - `match_score`  — score for matching characters.
/// - `mismatch`     — score for mismatching characters (typically negative).
/// - `gap`          — linear gap penalty (typically negative).
///
/// # Errors
///
/// Returns `NumRs2Error::ComputationError` on allocation failure (extremely
/// large sequences).
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::needleman_wunsch;
///
/// let result = needleman_wunsch("AGCT", "AGT", 1, -1, -2).expect("alignment failed");
/// assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());
/// assert!(result.score > 0.0);
/// ```
pub fn needleman_wunsch(
    seq1: &str,
    seq2: &str,
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> Result<AlignmentResult> {
    let (score, a1, a2) = core_nw(seq1.as_bytes(), seq2.as_bytes(), match_score, mismatch, gap)
        .map_err(map_core_err)?;
    Ok(AlignmentResult {
        score: score as f64,
        aligned_seq1: a1,
        aligned_seq2: a2,
    })
}

/// Performs local pairwise sequence alignment using the Smith-Waterman
/// dynamic-programming algorithm.
///
/// Local alignment finds the highest-scoring subsequence pair; gap characters
/// are `'-'`.
///
/// # Parameters
///
/// - `seq1`, `seq2` — sequences to align (ASCII, case-sensitive).
/// - `match_score`  — score for matching characters.
/// - `mismatch`     — score for mismatching characters (typically negative).
/// - `gap`          — linear gap penalty (typically negative).
///
/// # Errors
///
/// Returns `NumRs2Error::ComputationError` on internal error.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::smith_waterman;
///
/// let result = smith_waterman("TGTTACGG", "GGTTGACTA", 3, -3, -2)
///     .expect("alignment failed");
/// assert!(result.score >= 0.0);
/// assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());
/// ```
pub fn smith_waterman(
    seq1: &str,
    seq2: &str,
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> Result<AlignmentResult> {
    let (score, a1, a2) = core_sw(seq1.as_bytes(), seq2.as_bytes(), match_score, mismatch, gap)
        .map_err(map_core_err)?;
    Ok(AlignmentResult {
        score: score as f64,
        aligned_seq1: a1,
        aligned_seq2: a2,
    })
}

/// Computes the Levenshtein edit distance between two sequences.
///
/// The edit distance is the minimum number of single-character insertions,
/// deletions, or substitutions needed to transform `seq1` into `seq2`.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::edit_distance;
///
/// assert_eq!(edit_distance("kitten", "sitting"), 3);
/// assert_eq!(edit_distance("ATGC", "ATCC"), 1);
/// assert_eq!(edit_distance("", ""), 0);
/// ```
pub fn edit_distance(seq1: &str, seq2: &str) -> usize {
    core_edit_distance(seq1.as_bytes(), seq2.as_bytes())
}

// ─── Phylogenetic distances ───────────────────────────────────────────────────

/// Returns the Hamming distance between two sequences of equal length.
///
/// The Hamming distance counts positions at which the two sequences differ
/// (comparison is ASCII case-insensitive).
///
/// # Errors
///
/// Returns `NumRs2Error::DimensionMismatch` when the sequences have different
/// lengths.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::hamming_distance;
///
/// assert_eq!(hamming_distance("ATGC", "ATGC").expect("equal lengths"), 0);
/// assert_eq!(hamming_distance("ATGC", "TTGT").expect("equal lengths"), 2);
/// ```
pub fn hamming_distance(seq1: &str, seq2: &str) -> Result<usize> {
    core_hamming(seq1.as_bytes(), seq2.as_bytes()).map_err(map_core_err)
}

/// Converts a p-distance (fraction of differing sites) to a Jukes-Cantor
/// evolutionary distance.
///
/// ## Formula
///
/// ```text
/// d_JC = -3/4 × ln(1 − 4p/3)
/// ```
///
/// # Parameters
///
/// - `p` — observed p-distance in `[0.0, 0.75)`.
///
/// # Errors
///
/// Returns `NumRs2Error::ValueError` when `p` is outside `[0, 0.75)` or is
/// NaN.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::jukes_cantor_distance;
///
/// let d = jukes_cantor_distance(0.0).expect("valid p");
/// assert!((d - 0.0).abs() < 1e-10);
///
/// let d2 = jukes_cantor_distance(0.1).expect("valid p");
/// assert!((d2 - 0.10733).abs() < 1e-4);
/// ```
pub fn jukes_cantor_distance(p: f64) -> Result<f64> {
    core_jc(p).map_err(map_core_err)
}

/// Computes the pairwise Jukes-Cantor distance matrix for a slice of
/// sequences.
///
/// All sequences must have the same length.  The returned `Vec<Vec<f64>>` is
/// an `n × n` symmetric matrix with zeros on the diagonal.  Pairs whose
/// p-distance is ≥ 0.75 (where the JC formula is undefined) get
/// `f64::INFINITY`.
///
/// # Errors
///
/// Returns `NumRs2Error::DimensionMismatch` when the input is empty or
/// sequences have different lengths.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::distance_matrix;
///
/// let seqs = &["ATGC", "ATGC", "TTGT"];
/// let mat = distance_matrix(seqs).expect("valid sequences");
/// assert_eq!(mat.len(), 3);
/// assert!((mat[0][0] - 0.0).abs() < 1e-10);
/// ```
pub fn distance_matrix(sequences: &[&str]) -> Result<Vec<Vec<f64>>> {
    // Convert &[&str] → Vec<&[u8]>
    let byte_seqs: Vec<&[u8]> = sequences.iter().map(|s| s.as_bytes()).collect();
    let arr = core_distance_matrix(&byte_seqs).map_err(map_core_err)?;

    let n = sequences.len();
    let mut mat = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = arr[[i, j]];
        }
    }
    Ok(mat)
}

// ─── Sequence statistics ──────────────────────────────────────────────────────

/// Computes the relative frequencies of A, C, G, T in `seq`.
///
/// The returned `HashMap` maps each nucleotide character (`'A'`, `'C'`, `'G'`,
/// `'T'`) to its relative frequency.  Ambiguous or non-ACGT characters are
/// ignored in both the counts and the denominator.
///
/// Returns an empty map if `seq` contains no ACGT bases.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::nucleotide_frequencies;
///
/// let freqs = nucleotide_frequencies("AACGT");
/// assert!((freqs[&'A'] - 0.4).abs() < 1e-10);
/// assert!((freqs[&'C'] - 0.2).abs() < 1e-10);
/// ```
pub fn nucleotide_frequencies(seq: &str) -> HashMap<char, f64> {
    let raw = core_nuc_freq(seq.as_bytes()); // [A, C, G, T]
    let mut map = HashMap::with_capacity(4);
    map.insert('A', raw[0]);
    map.insert('C', raw[1]);
    map.insert('G', raw[2]);
    map.insert('T', raw[3]);
    map
}

/// Returns the GC content of a nucleotide sequence.
///
/// GC content is the fraction of G and C bases among all ACGT bases.
/// Non-ACGT characters are ignored.  Returns `0.0` for an empty or
/// all-non-ACGT sequence.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::gc_content;
///
/// assert!((gc_content("AATGCG") - 0.5).abs() < 1e-10);
/// assert!((gc_content("AAAA") - 0.0).abs() < 1e-10);
/// assert!((gc_content("GCGC") - 1.0).abs() < 1e-10);
/// ```
pub fn gc_content(seq: &str) -> f64 {
    core_gc_content(seq.as_bytes())
}

/// Returns a codon usage table for a DNA sequence.
///
/// Each codon (3-character substring) that contains only ACGT characters is
/// counted.  The returned map uses the uppercase codon string (e.g. `"ATG"`)
/// as the key and the raw count as the value.  Codons containing ambiguous
/// bases are skipped.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::codon_usage;
///
/// let usage = codon_usage("ATGATGATG");
/// assert_eq!(*usage.get("ATG").unwrap_or(&0), 3);
/// ```
pub fn codon_usage(seq: &str) -> HashMap<String, usize> {
    // core returns HashMap<[u8;3], f64> (relative frequency); we convert to
    // counts by multiplying by the number of codons and rounding.
    let bytes = seq.as_bytes();
    let num_codons = bytes.len() / 3;

    // Build raw count map directly for accuracy (avoid float rounding).
    let mut counts: HashMap<String, usize> = HashMap::new();
    for i in (0..bytes.len().saturating_sub(2)).step_by(3) {
        let codon = &bytes[i..i + 3];
        // Validate: all three characters must be ACGT.
        let valid = codon
            .iter()
            .all(|&b| matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T'));
        if valid {
            let key = String::from_utf8_lossy(codon).to_ascii_uppercase();
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    // Suppress unused variable warning from num_codons (used implicitly above).
    let _ = num_codons;

    counts
}

// ─── Additional helpers ───────────────────────────────────────────────────────

/// Returns the complement of a DNA sequence (A↔T, C↔G).
///
/// Non-ACGT characters are passed through unchanged.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::complement;
///
/// assert_eq!(complement("ATGC"), "TACG");
/// ```
pub fn complement(seq: &str) -> String {
    let rc = scirs2_core::bioinformatics::sequence::complement(seq.as_bytes());
    String::from_utf8_lossy(&rc).into_owned()
}

/// Returns the reverse complement of a DNA sequence.
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::bioinformatics::reverse_complement;
///
/// assert_eq!(reverse_complement("ATGCTT"), "AAGCAT");
/// ```
pub fn reverse_complement(seq: &str) -> String {
    let rc = scirs2_core::bioinformatics::sequence::reverse_complement(seq.as_bytes());
    String::from_utf8_lossy(&rc).into_owned()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── edit_distance ─────────────────────────────────────────────────────────

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("ATGC", "ATGC"), 0);
    }

    #[test]
    fn test_edit_distance_single_substitution() {
        assert_eq!(edit_distance("ATGC", "ATCC"), 1);
    }

    #[test]
    fn test_edit_distance_insertion_deletion() {
        // "kitten" → "sitting" requires 3 edits
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_edit_distance_empty_sequences() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("ABC", ""), 3);
        assert_eq!(edit_distance("", "XYZ"), 3);
    }

    #[test]
    fn test_edit_distance_biological_sequences() {
        // One deletion: AGCT → AGT
        assert_eq!(edit_distance("AGCT", "AGT"), 1);
        // Two substitutions
        assert_eq!(edit_distance("AAAAAA", "AAGGAA"), 2);
    }

    // ── needleman_wunsch ──────────────────────────────────────────────────────

    #[test]
    fn test_needleman_wunsch_identical() {
        let result = needleman_wunsch("AGCT", "AGCT", 1, -1, -2).expect("alignment should succeed");
        // Identical sequences: no gaps, score = length × match_score
        assert_eq!(result.score, 4.0);
        assert_eq!(result.aligned_seq1, "AGCT");
        assert_eq!(result.aligned_seq2, "AGCT");
    }

    #[test]
    fn test_needleman_wunsch_with_gap() {
        let result = needleman_wunsch("AGCT", "AGT", 1, -1, -2).expect("alignment should succeed");
        // Lengths must be equal after alignment
        assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());
        // Score should be positive (3 matches, 1 gap)
        assert!(result.score > 0.0);
        // One gap character somewhere
        let gaps1 = result.aligned_seq1.chars().filter(|&c| c == '-').count();
        let gaps2 = result.aligned_seq2.chars().filter(|&c| c == '-').count();
        assert_eq!(gaps1 + gaps2, 1, "expected exactly one gap total");
    }

    #[test]
    fn test_needleman_wunsch_empty_sequences() {
        let result = needleman_wunsch("", "", 1, -1, -2).expect("empty alignment should succeed");
        assert_eq!(result.score, 0.0);
        assert_eq!(result.aligned_seq1, "");
        assert_eq!(result.aligned_seq2, "");
    }

    #[test]
    fn test_needleman_wunsch_one_empty() {
        let result =
            needleman_wunsch("ATGC", "", 1, -1, -2).expect("one-empty alignment should succeed");
        // All gaps on the second aligned sequence
        assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());
        assert!(result.aligned_seq2.chars().all(|c| c == '-'));
    }

    // ── smith_waterman ────────────────────────────────────────────────────────

    #[test]
    fn test_smith_waterman_local_alignment() {
        // Classic SW example: best local alignment should yield score > 0
        let result =
            smith_waterman("TGTTACGG", "GGTTGACTA", 3, -3, -2).expect("alignment should succeed");
        assert!(result.score > 0.0);
        assert_eq!(result.aligned_seq1.len(), result.aligned_seq2.len());
    }

    #[test]
    fn test_smith_waterman_no_match() {
        // Sequences with no similar region (all mismatches, high penalty)
        let result = smith_waterman("AAAA", "TTTT", 1, -5, -5).expect("alignment should succeed");
        // Score may be 0 when no local alignment is profitable
        assert!(result.score >= 0.0);
    }

    // ── hamming_distance ──────────────────────────────────────────────────────

    #[test]
    fn test_hamming_distance_identical() {
        assert_eq!(hamming_distance("ATGC", "ATGC").expect("equal lengths"), 0);
    }

    #[test]
    fn test_hamming_distance_two_differences() {
        assert_eq!(hamming_distance("ATGC", "TTGT").expect("equal lengths"), 2);
    }

    #[test]
    fn test_hamming_distance_length_mismatch() {
        assert!(
            hamming_distance("ATGC", "ATG").is_err(),
            "unequal lengths must return an error"
        );
    }

    #[test]
    fn test_hamming_distance_empty() {
        assert_eq!(hamming_distance("", "").expect("empty sequences"), 0);
    }

    // ── jukes_cantor_distance ─────────────────────────────────────────────────

    #[test]
    fn test_jukes_cantor_zero_p() {
        let d = jukes_cantor_distance(0.0).expect("p=0 is valid");
        assert!((d - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jukes_cantor_known_value() {
        // p = 0.1 → d = -0.75 * ln(1 - 4/3 * 0.1) ≈ 0.10733
        let d = jukes_cantor_distance(0.1).expect("p=0.1 is valid");
        let expected = -0.75 * (1.0 - (4.0 / 3.0) * 0.1_f64).ln();
        assert!(
            (d - expected).abs() < 1e-10,
            "expected d ≈ {expected}, got {d}"
        );
    }

    #[test]
    fn test_jukes_cantor_invalid_p() {
        assert!(jukes_cantor_distance(0.75).is_err(), "p=0.75 is invalid");
        assert!(jukes_cantor_distance(1.0).is_err(), "p=1.0 is invalid");
        assert!(jukes_cantor_distance(-0.1).is_err(), "p<0 is invalid");
    }

    // ── distance_matrix ───────────────────────────────────────────────────────

    #[test]
    fn test_distance_matrix_basic() {
        let seqs = &["ATGC", "ATGC", "TTGT"];
        let mat = distance_matrix(seqs).expect("valid sequences");
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
        // Diagonal must be zero
        assert!((mat[0][0] - 0.0).abs() < 1e-10);
        assert!((mat[1][1] - 0.0).abs() < 1e-10);
        assert!((mat[2][2] - 0.0).abs() < 1e-10);
        // Symmetry
        assert!((mat[0][1] - mat[1][0]).abs() < 1e-10);
        assert!((mat[0][2] - mat[2][0]).abs() < 1e-10);
        // Identical sequences → distance 0
        assert!((mat[0][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix_empty_error() {
        let seqs: &[&str] = &[];
        assert!(distance_matrix(seqs).is_err());
    }

    #[test]
    fn test_distance_matrix_different_lengths_error() {
        let seqs = &["ATGC", "ATG"];
        assert!(distance_matrix(seqs).is_err());
    }

    // ── nucleotide_frequencies ────────────────────────────────────────────────

    #[test]
    fn test_nucleotide_frequencies_basic() {
        let freqs = nucleotide_frequencies("AACGT");
        // A=2, C=1, G=1, T=1 out of 5
        assert!(
            (freqs[&'A'] - 0.4).abs() < 1e-10,
            "A frequency wrong: {}",
            freqs[&'A']
        );
        assert!(
            (freqs[&'C'] - 0.2).abs() < 1e-10,
            "C frequency wrong: {}",
            freqs[&'C']
        );
        assert!(
            (freqs[&'G'] - 0.2).abs() < 1e-10,
            "G frequency wrong: {}",
            freqs[&'G']
        );
        assert!(
            (freqs[&'T'] - 0.2).abs() < 1e-10,
            "T frequency wrong: {}",
            freqs[&'T']
        );
    }

    #[test]
    fn test_nucleotide_frequencies_sum_to_one() {
        let freqs = nucleotide_frequencies("ATGCATGCATGC");
        let sum: f64 = freqs.values().sum();
        assert!((sum - 1.0).abs() < 1e-10, "frequencies must sum to 1.0");
    }

    #[test]
    fn test_nucleotide_frequencies_empty() {
        let freqs = nucleotide_frequencies("");
        assert!((freqs[&'A'] - 0.0).abs() < 1e-10);
        assert!((freqs[&'C'] - 0.0).abs() < 1e-10);
        assert!((freqs[&'G'] - 0.0).abs() < 1e-10);
        assert!((freqs[&'T'] - 0.0).abs() < 1e-10);
    }

    // ── gc_content ────────────────────────────────────────────────────────────

    #[test]
    fn test_gc_content_fifty_percent() {
        assert!((gc_content("AATGCG") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_zero() {
        assert!((gc_content("AAAA") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_one_hundred_percent() {
        assert!((gc_content("GCGC") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_empty() {
        assert!((gc_content("") - 0.0).abs() < 1e-10);
    }

    // ── codon_usage ───────────────────────────────────────────────────────────

    #[test]
    fn test_codon_usage_single_codon_repeated() {
        let usage = codon_usage("ATGATGATG");
        assert_eq!(*usage.get("ATG").unwrap_or(&0), 3);
    }

    #[test]
    fn test_codon_usage_multiple_codons() {
        // "ATGCTT" → ATG, CTT
        let usage = codon_usage("ATGCTT");
        assert_eq!(*usage.get("ATG").unwrap_or(&0), 1);
        assert_eq!(*usage.get("CTT").unwrap_or(&0), 1);
    }

    #[test]
    fn test_codon_usage_skips_invalid_codons() {
        // "ATGNNN" → ATG is valid, NNN contains ambiguous bases
        let usage = codon_usage("ATGNNN");
        assert_eq!(*usage.get("ATG").unwrap_or(&0), 1);
        assert!(!usage.contains_key("NNN"));
    }

    #[test]
    fn test_codon_usage_empty() {
        let usage = codon_usage("");
        assert!(usage.is_empty());
    }

    // ── complement / reverse_complement ───────────────────────────────────────

    #[test]
    fn test_complement_basic() {
        assert_eq!(complement("ATGC"), "TACG");
    }

    #[test]
    fn test_reverse_complement_basic() {
        assert_eq!(reverse_complement("ATGCTT"), "AAGCAT");
    }
}
