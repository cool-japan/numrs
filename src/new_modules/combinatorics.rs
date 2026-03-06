//! Combinatorics Module for NumRS2
//!
//! This module provides a NumPy-compatible combinatorial mathematics API implemented
//! natively in pure Rust, leveraging algorithmic approaches consistent with the
//! `scirs2-core` 0.3.0 combinatorics infrastructure.
//!
//! # Overview
//!
//! The module is organized into four main areas:
//!
//! - **Sequences**: Integer sequences such as Fibonacci, Catalan, Bell, and Stirling numbers
//! - **Permutations**: Counting and enumeration of permutations and combinations
//! - **Partitions**: Integer partition counting and enumeration
//! - **Number Theory**: GCD, LCM, primality testing, and prime factorization
//!
//! # Design Philosophy
//!
//! All functions that may produce values exceeding `u64::MAX` return `Option<u64>` using
//! checked arithmetic internally — no `unwrap()` appears in production paths. Functions
//! whose outputs are mathematically bounded (e.g., `gcd`) return plain values.
//!
//! # Mathematical References
//!
//! - Knuth, D. E. (2011). *The Art of Computer Programming*, Vol. 4A.
//! - OEIS: The On-Line Encyclopedia of Integer Sequences, <https://oeis.org/>
//! - Graham, Knuth, Patashnik (1994). *Concrete Mathematics*.

// ─────────────────────────────────────────────────────────────────────────────
// Sequences
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the *n*-th Fibonacci number F(n) using a simple iterative method.
///
/// The Fibonacci sequence is defined by:
///
/// ```text
/// F(0) = 0,  F(1) = 1,  F(n) = F(n-1) + F(n-2)   for n ≥ 2
/// ```
///
/// Returns `None` if the result would overflow `u64`. The largest representable
/// value is F(93) = 12 200 160 415 121 876 738; F(94) overflows.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::fibonacci;
/// assert_eq!(fibonacci(0), Some(0));
/// assert_eq!(fibonacci(1), Some(1));
/// assert_eq!(fibonacci(10), Some(55));
/// assert!(fibonacci(94).is_none());
/// ```
pub fn fibonacci(n: u64) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 1..n {
        let c = a.checked_add(b)?;
        a = b;
        b = c;
    }
    Some(b)
}

/// Build a vector containing the first *n* Fibonacci numbers: [F(0), F(1), …, F(n-1)].
///
/// Values that would overflow `u64` are silently omitted; the returned vector
/// may therefore be shorter than *n* for very large requests.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::fibonacci_sequence;
/// assert_eq!(fibonacci_sequence(8), vec![0, 1, 1, 2, 3, 5, 8, 13]);
/// ```
pub fn fibonacci_sequence(n: usize) -> Vec<u64> {
    // Use the scalar fibonacci() function to produce each element.
    // This is O(n^2) but avoids all subtle iterative state issues.
    // For n ≤ 94 this is entirely acceptable.
    let mut result = Vec::with_capacity(n.min(94));
    for i in 0..n {
        match fibonacci(i as u64) {
            Some(v) => result.push(v),
            None => break,
        }
    }
    result
}

/// Compute the *n*-th Catalan number C(n).
///
/// The Catalan numbers count, among many other things, the number of valid
/// parenthesizations of n+1 factors.  They satisfy:
///
/// ```text
/// C(0) = 1,  C(n) = C(2n, n) / (n+1)
/// ```
///
/// Returns `None` on `u64` overflow. C(35) is the last value that fits.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::catalan_number;
/// assert_eq!(catalan_number(0), Some(1));
/// assert_eq!(catalan_number(5), Some(42));
/// ```
pub fn catalan_number(n: u64) -> Option<u64> {
    // Use the recurrence C(n) = C(n-1) * 2(2n-1) / (n+1) to avoid large binomials.
    let mut c: u64 = 1;
    for i in 0..n {
        // c = c * 2*(2i+1) / (i+2)
        let num = c.checked_mul(2 * (2 * i + 1))?;
        c = num / (i + 2);
    }
    Some(c)
}

/// Compute the *n*-th Bell number B(n).
///
/// Bell numbers count the number of partitions of a set of *n* elements.
/// The first few values are: 1, 1, 2, 5, 15, 52, 203, 877, …
///
/// Returns `None` if the result overflows `u64`. B(20) = 51 724 158 235 372 is
/// representable; values beyond B(23) overflow.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::bell_number;
/// assert_eq!(bell_number(0), Some(1));
/// assert_eq!(bell_number(4), Some(15));
/// ```
pub fn bell_number(n: u32) -> Option<u64> {
    // Build Bell's triangle iteratively.
    // Row 0: [1]
    // Row k: first element = last element of row k-1; remaining elements = partial sums.
    let n = n as usize;
    let mut row = vec![1u64];
    for _ in 0..n {
        let mut next_row = Vec::with_capacity(row.len() + 1);
        next_row.push(*row.last()?);
        for j in 0..row.len() {
            let val = next_row[j].checked_add(row[j])?;
            next_row.push(val);
        }
        row = next_row;
    }
    row.first().copied()
}

/// Compute the Stirling number of the second kind S(n, k).
///
/// S(n, k) counts the number of ways to partition a set of *n* elements into
/// exactly *k* non-empty subsets.  The explicit formula is:
///
/// ```text
/// S(n, k) = (1/k!) * Σ_{j=0}^{k} (-1)^{k-j} * C(k,j) * j^n
/// ```
///
/// Returns `None` on `u64` overflow.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::stirling_second;
/// assert_eq!(stirling_second(4, 2), Some(7));
/// assert_eq!(stirling_second(5, 3), Some(25));
/// ```
pub fn stirling_second(n: u32, k: u32) -> Option<u64> {
    let n = n as usize;
    let k = k as usize;
    if k > n {
        return Some(0);
    }
    if k == 0 {
        return if n == 0 { Some(1) } else { Some(0) };
    }
    // Dynamic programming: S(i, j) = j * S(i-1, j) + S(i-1, j-1)
    // Use a 1-D rolling array of length k+1.
    let mut dp = vec![0u64; k + 1];
    dp[0] = 1; // S(0,0) = 1
    for _i in 1..=n {
        // Iterate j from k down to 1 to avoid using updated values.
        for j in (1..=k).rev() {
            let j_times = (j as u64).checked_mul(dp[j])?;
            dp[j] = j_times.checked_add(dp[j - 1])?;
        }
        dp[0] = 0; // S(i, 0) = 0 for i > 0
    }
    Some(dp[k])
}

// ─────────────────────────────────────────────────────────────────────────────
// Permutations
// ─────────────────────────────────────────────────────────────────────────────

/// Compute n! (factorial) using checked arithmetic.
///
/// Returns `None` if the result overflows `u64`. The largest representable
/// factorial is 20! = 2 432 902 008 176 640 000; 21! overflows.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::factorial;
/// assert_eq!(factorial(0), Some(1));
/// assert_eq!(factorial(10), Some(3628800));
/// assert!(factorial(21).is_none());
/// ```
pub fn factorial(n: u64) -> Option<u64> {
    let mut result: u64 = 1;
    for i in 2..=n {
        result = result.checked_mul(i)?;
    }
    Some(result)
}

/// Compute the number of *k*-permutations of *n* elements: P(n, k) = n! / (n-k)!
///
/// This counts ordered selections of *k* items from *n* distinct items.
///
/// Returns `None` if k > n or if the result overflows `u64`.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::permutation_count;
/// assert_eq!(permutation_count(5, 3), Some(60));
/// assert_eq!(permutation_count(0, 0), Some(1));
/// assert_eq!(permutation_count(3, 5), None); // k > n
/// ```
pub fn permutation_count(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return None;
    }
    // P(n,k) = n * (n-1) * … * (n-k+1)  — multiply k factors
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.checked_mul(n - i)?;
    }
    Some(result)
}

/// Compute the binomial coefficient C(n, k) = n! / (k! * (n-k)!)
///
/// Uses a multiplicative formula with early overflow detection.
/// Returns `None` if the result overflows `u64`.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::combination_count;
/// assert_eq!(combination_count(5, 2), Some(10));
/// assert_eq!(combination_count(10, 0), Some(1));
/// assert_eq!(combination_count(3, 5), Some(0)); // k > n → 0
/// ```
pub fn combination_count(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    // Symmetry: use min(k, n-k) to minimise the number of multiplications.
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        // result = result * (n - i) / (i + 1)
        // Because C(n,k) is always an integer, the division is exact at each step
        // when we multiply before dividing (using the standard iterative formula).
        result = result.checked_mul(n - i)?;
        result /= i + 1;
    }
    Some(result)
}

/// Advance a mutable permutation slice to the next lexicographic permutation.
///
/// Returns `true` if the permutation was advanced, or `false` if the slice
/// was already the last permutation (in descending order) and has been reset
/// to the first (ascending) permutation.
///
/// # Algorithm
///
/// Uses the classic algorithm (Knuth Vol. 4A, Algorithm L):
/// 1. Find the largest index *i* such that `perm[i] < perm[i+1]`.
/// 2. Find the largest index *j* such that `perm[i] < perm[j]`.
/// 3. Swap `perm[i]` and `perm[j]`, then reverse the suffix `perm[i+1..]`.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::next_permutation;
/// let mut p = vec![0usize, 1, 2];
/// assert!(next_permutation(&mut p));
/// assert_eq!(p, [0, 2, 1]);
/// ```
pub fn next_permutation(perm: &mut [usize]) -> bool {
    let n = perm.len();
    if n < 2 {
        return false;
    }
    // Step 1: find the rightmost ascent.
    let mut i = n - 1;
    while i > 0 && perm[i - 1] >= perm[i] {
        i -= 1;
    }
    if i == 0 {
        // Already the last permutation; reverse to reset.
        perm.reverse();
        return false;
    }
    let pivot = i - 1;
    // Step 2: find the rightmost element greater than perm[pivot].
    let mut j = n - 1;
    while perm[j] <= perm[pivot] {
        j -= 1;
    }
    // Step 3: swap and reverse suffix.
    perm.swap(pivot, j);
    perm[pivot + 1..].reverse();
    true
}

/// Generate all permutations of [0, 1, …, n-1] in lexicographic order.
///
/// The returned vector has n! elements. For n ≥ 13 this grows very large;
/// prefer the iterator pattern via [`next_permutation`] for large n.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::all_permutations;
/// assert_eq!(all_permutations(3).len(), 6);
/// ```
pub fn all_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let count = factorial(n as u64).unwrap_or(u64::MAX) as usize;
    let mut result = Vec::with_capacity(count.min(40320)); // cap at 8! for safety
    let mut perm: Vec<usize> = (0..n).collect();
    loop {
        result.push(perm.clone());
        if !next_permutation(&mut perm) {
            break;
        }
    }
    result
}

/// Generate all *k*-element combinations of [0, 1, …, n-1] in lexicographic order.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::all_combinations;
/// let combs = all_combinations(4, 2);
/// assert_eq!(combs.len(), 6);
/// assert_eq!(combs[0], vec![0, 1]);
/// ```
pub fn all_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k > n {
        return Vec::new();
    }
    if k == 0 {
        return vec![vec![]];
    }
    let count = combination_count(n as u64, k as u64).unwrap_or(0) as usize;
    let mut result = Vec::with_capacity(count);

    // Iterative combination generation using the revolving-door algorithm index.
    let mut combo: Vec<usize> = (0..k).collect();
    loop {
        result.push(combo.clone());
        // Find the rightmost element that can be incremented.
        let mut i = k;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if combo[i] < n - k + i {
                break;
            }
        }
        combo[i] += 1;
        for j in i + 1..k {
            combo[j] = combo[j - 1] + 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Partitions
// ─────────────────────────────────────────────────────────────────────────────

/// Count the number of integer partitions of *n*.
///
/// An integer partition of *n* is a way of writing *n* as a sum of positive
/// integers where order does not matter.  The partition function p(n) grows
/// rapidly; p(100) = 190 569 292.
///
/// Returns `None` on `u64` overflow (occurs around n = 415).
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::partition_count;
/// assert_eq!(partition_count(0), Some(1));
/// assert_eq!(partition_count(5), Some(7));
/// assert_eq!(partition_count(10), Some(42));
/// ```
pub fn partition_count(n: usize) -> Option<u64> {
    // Euler's pentagonal number theorem recurrence:
    // p(n) = Σ_{k≠0} (-1)^{k+1} * p(n - k*(3k-1)/2)
    // where k ranges over ±1, ±2, ±3, …
    let mut p = vec![0u64; n + 1];
    p[0] = 1;
    for i in 1..=n {
        let mut k: i64 = 1;
        loop {
            let pent_pos = (k * (3 * k - 1) / 2) as usize;
            if pent_pos > i {
                break;
            }
            let sign_pos = k % 2 == 1;
            // Positive pentagonal index
            if sign_pos {
                p[i] = p[i].checked_add(p[i - pent_pos])?;
            } else {
                p[i] = p[i].saturating_sub(p[i - pent_pos]);
                // Subtraction underflow would mean a programming error; treat as 0.
            }

            // Negative pentagonal index: k*(3k+1)/2
            let pent_neg = (k * (3 * k + 1) / 2) as usize;
            if pent_neg <= i {
                if sign_pos {
                    p[i] = p[i].checked_add(p[i - pent_neg])?;
                } else {
                    p[i] = p[i].saturating_sub(p[i - pent_neg]);
                }
            }
            k += 1;
        }
    }
    Some(p[n])
}

/// Enumerate all integer partitions of *n* in lexicographically descending order.
///
/// Each partition is represented as a `Vec<usize>` of parts sorted from largest
/// to smallest (the standard mathematical convention).
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::enumerate_partitions;
/// let parts = enumerate_partitions(4);
/// // [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
/// assert_eq!(parts.len(), 5);
/// ```
pub fn enumerate_partitions(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let mut partition = Vec::with_capacity(n);
    enumerate_partitions_helper(n, n, &mut partition, &mut result);
    result
}

/// Recursive helper for [`enumerate_partitions`].
///
/// Generates all partitions of `remaining` where the largest part is at most `max_part`.
fn enumerate_partitions_helper(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }
    let upper = remaining.min(max_part);
    for part in (1..=upper).rev() {
        current.push(part);
        enumerate_partitions_helper(remaining - part, part, current, result);
        current.pop();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Number Theory
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the greatest common divisor of *a* and *b* using the Euclidean algorithm.
///
/// By convention gcd(0, 0) = 0.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::gcd;
/// assert_eq!(gcd(48, 18), 6);
/// assert_eq!(gcd(0, 7), 7);
/// ```
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute the least common multiple of *a* and *b*.
///
/// Returns `None` if the result overflows `u64`.  lcm(0, n) = 0 by convention.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::lcm;
/// assert_eq!(lcm(4, 6), Some(12));
/// assert_eq!(lcm(0, 5), Some(0));
/// ```
pub fn lcm(a: u64, b: u64) -> Option<u64> {
    if a == 0 || b == 0 {
        return Some(0);
    }
    let g = gcd(a, b);
    // Divide before multiplying to reduce overflow risk.
    (a / g).checked_mul(b)
}

/// Test whether *n* is prime using trial division up to √n.
///
/// Returns `false` for n < 2. For large *n*, consider a Miller-Rabin variant.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::is_prime;
/// assert!(is_prime(2));
/// assert!(is_prime(97));
/// assert!(!is_prime(1));
/// assert!(!is_prime(100));
/// ```
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    // Check divisors of the form 6k ± 1 up to √n.
    let mut i: u64 = 5;
    while i * i <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

/// Return the prime factorization of *n* as a sorted vector of prime factors (with repeats).
///
/// Returns an empty vector for n = 0 or n = 1. The factors appear in non-decreasing order.
///
/// # Examples
///
/// ```rust
/// use numrs::new_modules::combinatorics::prime_factors;
/// assert_eq!(prime_factors(12), vec![2, 2, 3]);
/// assert_eq!(prime_factors(13), vec![13]);
/// assert_eq!(prime_factors(1), vec![]);
/// ```
pub fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    if n <= 1 {
        return factors;
    }
    // Extract all factors of 2.
    while n.is_multiple_of(2) {
        factors.push(2);
        n /= 2;
    }
    // Extract odd prime factors starting at 3.
    let mut d: u64 = 3;
    while d * d <= n {
        while n.is_multiple_of(d) {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }
    // If anything remains, it is a prime factor greater than √n.
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fibonacci ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fibonacci_base_cases() {
        assert_eq!(fibonacci(0), Some(0));
        assert_eq!(fibonacci(1), Some(1));
        assert_eq!(fibonacci(2), Some(1));
    }

    #[test]
    fn test_fibonacci_known_values() {
        // F(10) = 55, F(20) = 6765, F(30) = 832040
        assert_eq!(fibonacci(10), Some(55));
        assert_eq!(fibonacci(20), Some(6765));
        assert_eq!(fibonacci(30), Some(832040));
    }

    #[test]
    fn test_fibonacci_max_u64() {
        // F(93) is the largest Fibonacci number fitting in u64.
        assert_eq!(fibonacci(93), Some(12200160415121876738u64));
        // F(94) overflows.
        assert!(fibonacci(94).is_none());
    }

    #[test]
    fn test_fibonacci_sequence_length() {
        let seq = fibonacci_sequence(10);
        assert_eq!(seq.len(), 10);
        assert_eq!(seq[0], 0);
        assert_eq!(seq[1], 1);
        assert_eq!(seq[9], 34);
    }

    #[test]
    fn test_fibonacci_sequence_empty() {
        assert!(fibonacci_sequence(0).is_empty());
    }

    // ── Catalan ───────────────────────────────────────────────────────────────

    #[test]
    fn test_catalan_known_values() {
        let expected = [1u64, 1, 2, 5, 14, 42, 132, 429, 1430, 4862];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(catalan_number(i as u64), Some(exp), "C({i}) = {exp}");
        }
    }

    #[test]
    fn test_catalan_zero() {
        assert_eq!(catalan_number(0), Some(1));
    }

    // ── Bell ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_bell_known_values() {
        // B(0)..B(9): 1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147
        let expected = [1u64, 1, 2, 5, 15, 52, 203, 877, 4140, 21147];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(bell_number(i as u32), Some(exp), "B({i}) = {exp}");
        }
    }

    // ── Stirling Second Kind ──────────────────────────────────────────────────

    #[test]
    fn test_stirling_second_boundary() {
        assert_eq!(stirling_second(0, 0), Some(1));
        assert_eq!(stirling_second(1, 0), Some(0));
        assert_eq!(stirling_second(1, 1), Some(1));
        assert_eq!(stirling_second(2, 5), Some(0)); // k > n
    }

    #[test]
    fn test_stirling_second_known_values() {
        assert_eq!(stirling_second(3, 2), Some(3));
        assert_eq!(stirling_second(4, 2), Some(7));
        assert_eq!(stirling_second(4, 4), Some(1));
        assert_eq!(stirling_second(5, 3), Some(25));
    }

    // ── Factorial ─────────────────────────────────────────────────────────────

    #[test]
    fn test_factorial_base_cases() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
    }

    #[test]
    fn test_factorial_known_values() {
        assert_eq!(factorial(5), Some(120));
        assert_eq!(factorial(10), Some(3628800));
        assert_eq!(factorial(20), Some(2432902008176640000u64));
    }

    #[test]
    fn test_factorial_overflow() {
        assert!(factorial(21).is_none());
    }

    // ── Permutation count ─────────────────────────────────────────────────────

    #[test]
    fn test_permutation_count() {
        assert_eq!(permutation_count(5, 3), Some(60));
        assert_eq!(permutation_count(5, 0), Some(1));
        assert_eq!(permutation_count(5, 5), Some(120));
    }

    #[test]
    fn test_permutation_count_k_greater_than_n() {
        assert_eq!(permutation_count(3, 5), None);
    }

    // ── Combination count ─────────────────────────────────────────────────────

    #[test]
    fn test_combination_count_known() {
        assert_eq!(combination_count(5, 2), Some(10));
        assert_eq!(combination_count(10, 3), Some(120));
        assert_eq!(combination_count(0, 0), Some(1));
    }

    #[test]
    fn test_combination_count_k_greater_than_n() {
        assert_eq!(combination_count(3, 5), Some(0));
    }

    #[test]
    fn test_combination_count_symmetry() {
        // C(n,k) = C(n, n-k)
        assert_eq!(combination_count(10, 3), combination_count(10, 7));
    }

    // ── next_permutation ──────────────────────────────────────────────────────

    #[test]
    fn test_next_permutation_advance() {
        let mut p = vec![0usize, 1, 2];
        assert!(next_permutation(&mut p));
        assert_eq!(p, [0, 2, 1]);
    }

    #[test]
    fn test_next_permutation_last() {
        let mut p = vec![2usize, 1, 0];
        let advanced = next_permutation(&mut p);
        assert!(!advanced);
        // After wrapping, the slice is reset to ascending order.
        assert_eq!(p, [0, 1, 2]);
    }

    #[test]
    fn test_next_permutation_single_element() {
        let mut p = vec![42usize];
        assert!(!next_permutation(&mut p));
    }

    // ── all_permutations ──────────────────────────────────────────────────────

    #[test]
    fn test_all_permutations_count() {
        assert_eq!(all_permutations(0).len(), 1);
        assert_eq!(all_permutations(1).len(), 1);
        assert_eq!(all_permutations(3).len(), 6);
        assert_eq!(all_permutations(4).len(), 24);
    }

    #[test]
    fn test_all_permutations_content() {
        let perms = all_permutations(3);
        // First should be [0,1,2] and last [2,1,0] in lex order.
        assert_eq!(perms[0], vec![0, 1, 2]);
        assert_eq!(perms[5], vec![2, 1, 0]);
    }

    // ── all_combinations ──────────────────────────────────────────────────────

    #[test]
    fn test_all_combinations_count() {
        assert_eq!(all_combinations(4, 2).len(), 6);
        assert_eq!(all_combinations(5, 3).len(), 10);
        assert_eq!(all_combinations(5, 0).len(), 1);
        assert_eq!(all_combinations(3, 5).len(), 0);
    }

    #[test]
    fn test_all_combinations_content() {
        let combs = all_combinations(4, 2);
        assert_eq!(combs[0], vec![0, 1]);
        assert_eq!(combs[5], vec![2, 3]);
    }

    // ── partition_count ───────────────────────────────────────────────────────

    #[test]
    fn test_partition_count_known() {
        let expected = [1u64, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(partition_count(i), Some(exp), "p({i}) = {exp}");
        }
    }

    #[test]
    fn test_partition_count_larger() {
        assert_eq!(partition_count(20), Some(627));
        assert_eq!(partition_count(50), Some(204226));
    }

    // ── enumerate_partitions ──────────────────────────────────────────────────

    #[test]
    fn test_enumerate_partitions_zero() {
        let parts = enumerate_partitions(0);
        assert_eq!(parts, vec![vec![0usize; 0]]);
    }

    #[test]
    fn test_enumerate_partitions_four() {
        let parts = enumerate_partitions(4);
        // p(4) = 5
        assert_eq!(parts.len(), 5);
        // First partition is [4], last is [1,1,1,1].
        assert_eq!(parts[0], vec![4]);
        assert_eq!(parts[4], vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_enumerate_partitions_sums() {
        for n in 0..=8 {
            let parts = enumerate_partitions(n);
            for p in &parts {
                let s: usize = p.iter().sum();
                assert_eq!(s, n, "Partition {:?} should sum to {n}", p);
            }
        }
    }

    // ── GCD / LCM ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(0, 7), 7);
        assert_eq!(gcd(7, 0), 7);
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(13, 13), 13);
    }

    #[test]
    fn test_lcm_basic() {
        assert_eq!(lcm(4, 6), Some(12));
        assert_eq!(lcm(0, 5), Some(0));
        assert_eq!(lcm(7, 1), Some(7));
        assert_eq!(lcm(21, 6), Some(42));
    }

    // ── is_prime ──────────────────────────────────────────────────────────────

    #[test]
    fn test_is_prime_small() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
    }

    #[test]
    fn test_is_prime_larger() {
        assert!(is_prime(97));
        assert!(is_prime(7919));
        assert!(!is_prime(100));
        assert!(!is_prime(7918));
    }

    // ── prime_factors ─────────────────────────────────────────────────────────

    #[test]
    fn test_prime_factors_basic() {
        assert_eq!(prime_factors(1), Vec::<u64>::new());
        assert_eq!(prime_factors(12), vec![2, 2, 3]);
        assert_eq!(prime_factors(13), vec![13]);
        assert_eq!(prime_factors(360), vec![2, 2, 2, 3, 3, 5]);
    }

    #[test]
    fn test_prime_factors_product_equals_n() {
        for n in [2u64, 60, 97, 1024, 999983] {
            let factors = prime_factors(n);
            let product: u64 = factors.iter().product();
            assert_eq!(product, n, "prime_factors({n}) product mismatch");
        }
    }

    #[test]
    fn test_prime_factors_zero() {
        assert!(prime_factors(0).is_empty());
    }
}
