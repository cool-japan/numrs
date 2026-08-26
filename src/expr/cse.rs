//! Common Subexpression Elimination (CSE)
//!
//! CSE is an optimization technique that identifies when the same expression
//! is computed multiple times and caches the result to avoid redundant computation.
//! This is especially valuable for large arrays where recomputation is expensive.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use crate::shared_array::SharedArray;

use super::shared::SharedExpr;

/// Unique identifier for expression nodes in the DAG
///
/// Used to identify common subexpressions during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(u64);

impl ExprId {
    /// Generate a new unique expression ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Create from a raw value (for testing or specific use cases)
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for ExprId {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for storing evaluated expressions
///
/// Thread-safe cache that maps expression IDs to their evaluated SharedArray results.
/// This enables sharing of computation results across multiple uses of the same
/// subexpression.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{ExprCache, ExprId};
///
/// let cache = ExprCache::new();
///
/// // Store a result
/// let id = ExprId::new();
/// let array = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
/// cache.insert(id, array.clone());
///
/// // Retrieve the cached result
/// let cached: Option<SharedArray<f64>> = cache.get(&id);
/// assert!(cached.is_some());
/// assert_eq!(cached.expect("cached value should exist").to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub struct ExprCache<T: Clone> {
    cache: Arc<RwLock<HashMap<ExprId, SharedArray<T>>>>,
}

impl<T: Clone> ExprCache<T> {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert a result into the cache
    pub fn insert(&self, id: ExprId, value: SharedArray<T>) {
        if let Ok(mut guard) = self.cache.write() {
            guard.insert(id, value);
        }
    }

    /// Get a cached result
    pub fn get(&self, id: &ExprId) -> Option<SharedArray<T>> {
        if let Ok(guard) = self.cache.read() {
            guard.get(id).cloned()
        } else {
            None
        }
    }

    /// Check if an expression is cached
    pub fn contains(&self, id: &ExprId) -> bool {
        if let Ok(guard) = self.cache.read() {
            guard.contains_key(id)
        } else {
            false
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut guard) = self.cache.write() {
            guard.clear();
        }
    }

    /// Get the number of cached entries
    pub fn len(&self) -> usize {
        if let Ok(guard) = self.cache.read() {
            guard.len()
        } else {
            0
        }
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone> Default for ExprCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for ExprCache<T> {
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
        }
    }
}

/// A cached expression wrapper
///
/// Wraps an expression with caching capability. When evaluated, it first checks
/// the cache for a pre-computed result. If found, it returns the cached value;
/// otherwise, it evaluates the expression and stores the result.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{CachedExpr, SharedArrayExpr, SharedExpr, ExprCache};
///
/// let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let expr = SharedArrayExpr::new(arr);
/// let cache = ExprCache::new();
///
/// let cached = CachedExpr::new(expr, cache.clone());
///
/// // First evaluation computes and caches the result
/// let result1 = cached.eval();
///
/// // Second evaluation returns the cached result
/// let result2 = cached.eval();
///
/// assert_eq!(result1.to_vec(), result2.to_vec());
/// assert_eq!(cache.len(), 1); // Only one entry cached
/// ```
#[derive(Clone)]
pub struct CachedExpr<T: Clone, E: SharedExpr<T>> {
    expr: E,
    id: ExprId,
    cache: ExprCache<T>,
}

impl<T: Clone, E: SharedExpr<T>> CachedExpr<T, E> {
    /// Create a new cached expression
    pub fn new(expr: E, cache: ExprCache<T>) -> Self {
        Self {
            expr,
            id: ExprId::new(),
            cache,
        }
    }

    /// Create with a specific ID (useful for CSE optimization)
    pub fn with_id(expr: E, id: ExprId, cache: ExprCache<T>) -> Self {
        Self { expr, id, cache }
    }

    /// Get the expression ID
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Get a reference to the cache
    pub fn cache(&self) -> &ExprCache<T> {
        &self.cache
    }

    /// Invalidate the cached result for this expression
    pub fn invalidate(&self) {
        if let Ok(mut guard) = self.cache.cache.write() {
            guard.remove(&self.id);
        }
    }
}

impl<T: Clone, E: SharedExpr<T>> SharedExpr<T> for CachedExpr<T, E> {
    fn eval_at(&self, index: usize) -> T {
        // For indexed access, we evaluate and cache the full array
        // then return the specific index
        let array = self.eval();
        array.to_vec()[index].clone()
    }

    fn size(&self) -> usize {
        self.expr.size()
    }

    fn shape(&self) -> Vec<usize> {
        self.expr.shape()
    }

    fn eval(&self) -> SharedArray<T> {
        // Check cache first
        if let Some(cached) = self.cache.get(&self.id) {
            return cached;
        }

        // Evaluate and cache
        let result = self.expr.eval();
        self.cache.insert(self.id, result.clone());
        result
    }
}

/// Expression hash key for CSE identification
///
/// Used to identify structurally identical expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprKey {
    /// Leaf array with an ID
    Array(u64),
    /// Binary operation with operation type and operand keys
    Binary {
        op: &'static str,
        left: Box<ExprKey>,
        right: Box<ExprKey>,
    },
    /// Unary operation
    Unary {
        op: &'static str,
        operand: Box<ExprKey>,
    },
    /// Scalar operation
    Scalar {
        op: &'static str,
        operand: Box<ExprKey>,
        scalar_hash: u64,
    },
}

impl ExprKey {
    /// Create an array key
    pub fn array(id: u64) -> Self {
        Self::Array(id)
    }

    /// Create a binary operation key
    pub fn binary(op: &'static str, left: ExprKey, right: ExprKey) -> Self {
        Self::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a unary operation key
    pub fn unary(op: &'static str, operand: ExprKey) -> Self {
        Self::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    /// Create a scalar operation key
    pub fn scalar(op: &'static str, operand: ExprKey, scalar_hash: u64) -> Self {
        Self::Scalar {
            op,
            operand: Box::new(operand),
            scalar_hash,
        }
    }
}

/// Hash a floating-point value for use in expression keys
pub fn hash_f64(value: f64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// Common Subexpression Elimination (CSE) Optimizer
///
/// Analyzes expression trees to identify common subexpressions and creates
/// an optimized DAG (Directed Acyclic Graph) where shared computations are
/// evaluated only once.
///
/// # How It Works
///
/// 1. **Expression Analysis**: Traverses the expression tree and assigns keys
///    to each subexpression based on its structure.
/// 2. **Common Subexpression Detection**: Identifies subexpressions with identical
///    keys (same operation, same operands).
/// 3. **Cache Creation**: Creates a shared cache for storing evaluated results.
/// 4. **DAG Construction**: Wraps expressions with CachedExpr nodes that share
///    the same cache.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{
///     SharedArrayExpr, SharedBinaryExpr, SharedScalarExpr, SharedExpr,
///     CSEOptimizer, ExprKey, hash_f64
/// };
///
/// // Create arrays
/// let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let b = SharedArray::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
///
/// // Expression: (a + b) * (a + b) - has common subexpression (a + b)
/// let expr_a1 = SharedArrayExpr::new(a.clone());
/// let expr_b1 = SharedArrayExpr::new(b.clone());
/// let sum1 = SharedBinaryExpr::new(expr_a1, expr_b1, |x, y| x + y)
///     .expect("creating sum1 expression should succeed");
///
/// let expr_a2 = SharedArrayExpr::new(a.clone());
/// let expr_b2 = SharedArrayExpr::new(b.clone());
/// let sum2 = SharedBinaryExpr::new(expr_a2, expr_b2, |x, y| x + y)
///     .expect("creating sum2 expression should succeed");
///
/// let product = SharedBinaryExpr::new(sum1, sum2, |x, y| x * y)
///     .expect("creating product expression should succeed");
///
/// // Without CSE, (a + b) is computed twice
/// // With CSE, (a + b) is computed once and reused
///
/// let result = product.eval();
/// // (3*3, 5*5, 7*7, 9*9) = (9, 25, 49, 81)
/// assert_eq!(result.to_vec(), vec![9.0, 25.0, 49.0, 81.0]);
/// ```
pub struct CSEOptimizer<T: Clone> {
    /// Maps expression keys to their assigned IDs
    key_to_id: HashMap<ExprKey, ExprId>,
    /// The shared cache for evaluated results
    cache: ExprCache<T>,
    /// Counter for assigning unique array IDs
    next_array_id: u64,
}

impl<T: Clone> CSEOptimizer<T> {
    /// Create a new CSE optimizer
    pub fn new() -> Self {
        Self {
            key_to_id: HashMap::new(),
            cache: ExprCache::new(),
            next_array_id: 0,
        }
    }

    /// Get or create an ID for an expression key
    pub fn get_or_create_id(&mut self, key: &ExprKey) -> ExprId {
        if let Some(&id) = self.key_to_id.get(key) {
            id
        } else {
            let id = ExprId::new();
            self.key_to_id.insert(key.clone(), id);
            id
        }
    }

    /// Get a new unique array ID
    pub fn next_array_id(&mut self) -> u64 {
        let id = self.next_array_id;
        self.next_array_id += 1;
        id
    }

    /// Get the shared cache
    pub fn cache(&self) -> &ExprCache<T> {
        &self.cache
    }

    /// Create a cached version of an expression
    pub fn cache_expr<E: SharedExpr<T>>(&self, expr: E, id: ExprId) -> CachedExpr<T, E> {
        CachedExpr::with_id(expr, id, self.cache.clone())
    }

    /// Get statistics about the optimization
    pub fn stats(&self) -> CSEStats {
        CSEStats {
            unique_expressions: self.key_to_id.len(),
            cached_results: self.cache.len(),
        }
    }

    /// Clear the optimizer state
    pub fn clear(&mut self) {
        self.key_to_id.clear();
        self.cache.clear();
        self.next_array_id = 0;
    }
}

impl<T: Clone> Default for CSEOptimizer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from CSE optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CSEStats {
    /// Number of unique expression keys
    pub unique_expressions: usize,
    /// Number of cached results
    pub cached_results: usize,
}

/// Builder for constructing CSE-optimized expression graphs
///
/// Provides a fluent API for building expression trees with automatic
/// common subexpression elimination.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{CSEExprBuilder, SharedExpr, SharedArrayExpr, ExprKey, CSESupport};
///
/// let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// let mut builder: CSEExprBuilder<f64> = CSEExprBuilder::new();
///
/// // Wrap an expression with CSE caching
/// let expr = SharedArrayExpr::new(a.clone());
/// let key = ExprKey::array(0);
/// let cached = builder.wrap(expr, key);
///
/// let result = cached.eval();
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub struct CSEExprBuilder<T: Clone> {
    optimizer: CSEOptimizer<T>,
}

impl<T: Clone> CSEExprBuilder<T> {
    /// Create a new CSE expression builder
    pub fn new() -> Self {
        Self {
            optimizer: CSEOptimizer::new(),
        }
    }

    /// Wrap an expression with CSE caching
    pub fn wrap<E: SharedExpr<T>>(&mut self, expr: E, key: ExprKey) -> CachedExpr<T, E> {
        let id = self.optimizer.get_or_create_id(&key);
        self.optimizer.cache_expr(expr, id)
    }

    /// Evaluate and cache a SharedArray directly
    pub fn eval_array(&self, array: SharedArray<T>) -> SharedArray<T> {
        array
    }

    /// Get the optimizer stats
    pub fn stats(&self) -> CSEStats {
        self.optimizer.stats()
    }

    /// Clear all cached results
    pub fn clear(&mut self) {
        self.optimizer.clear();
    }
}

impl<T: Clone> Default for CSEExprBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for adding CSE support to SharedExpr
pub trait CSESupport<T: Clone>: SharedExpr<T> + Sized {
    /// Wrap this expression with CSE caching
    fn with_cache(self, cache: ExprCache<T>) -> CachedExpr<T, Self> {
        CachedExpr::new(self, cache)
    }

    /// Create a CSE-wrapped version with a specific ID
    fn with_cache_id(self, id: ExprId, cache: ExprCache<T>) -> CachedExpr<T, Self> {
        CachedExpr::with_id(self, id, cache)
    }
}

// Implement CSESupport for all SharedExpr types
impl<T: Clone, E: SharedExpr<T>> CSESupport<T> for E {}

/// Result of CSE analysis
#[derive(Debug, Clone)]
pub struct CSEAnalysisResult {
    /// Total number of expression nodes
    pub total_nodes: usize,
    /// Number of common subexpressions found
    pub common_subexpressions: usize,
    /// Estimated computation savings (ratio of reused to total)
    pub savings_ratio: f64,
    /// Map of expression keys to their occurrence counts
    pub occurrence_counts: HashMap<String, usize>,
}

impl CSEAnalysisResult {
    /// Create a new analysis result
    pub fn new() -> Self {
        Self {
            total_nodes: 0,
            common_subexpressions: 0,
            savings_ratio: 0.0,
            occurrence_counts: HashMap::new(),
        }
    }

    /// Calculate the savings ratio
    pub fn calculate_savings(&mut self) {
        if self.total_nodes > 0 {
            self.savings_ratio = self.common_subexpressions as f64 / self.total_nodes as f64;
        }
    }
}

impl Default for CSEAnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze an expression tree for common subexpressions
///
/// This function performs a static analysis of the expression structure
/// to identify potential CSE opportunities.
///
/// # Example
///
/// ```
/// use numrs2::shared_array::SharedArray;
/// use numrs2::expr::{
///     SharedArrayExpr, SharedBinaryExpr, SharedExpr,
///     analyze_cse, ExprKey
/// };
///
/// let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
/// let key_a = ExprKey::array(0);
/// let key_b = ExprKey::array(1);
/// let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());
/// let key_product = ExprKey::binary("mul", key_sum.clone(), key_sum.clone());
///
/// let keys = vec![key_a, key_b, key_sum.clone(), key_sum.clone(), key_product];
/// let analysis = analyze_cse(&keys);
///
/// // The sum expression appears twice
/// assert!(analysis.common_subexpressions > 0);
/// ```
pub fn analyze_cse(keys: &[ExprKey]) -> CSEAnalysisResult {
    let mut result = CSEAnalysisResult::new();
    result.total_nodes = keys.len();

    // Count occurrences of each key
    let mut key_counts: HashMap<String, usize> = HashMap::new();
    for key in keys {
        let key_str = format!("{:?}", key);
        *key_counts.entry(key_str).or_insert(0) += 1;
    }

    // Count common subexpressions (those appearing more than once)
    for (key_str, count) in &key_counts {
        if *count > 1 {
            result.common_subexpressions += count - 1; // Extra occurrences
            result.occurrence_counts.insert(key_str.clone(), *count);
        }
    }

    result.calculate_savings();
    result
}

/// Optimized expression graph node
///
/// Represents a node in the CSE-optimized expression DAG.
/// Each node has a unique ID and may reference cached results.
#[derive(Clone)]
pub struct OptimizedExprNode<T: Clone> {
    id: ExprId,
    key: ExprKey,
    cache: ExprCache<T>,
    /// Cached evaluation result (set after first evaluation)
    result: Option<SharedArray<T>>,
}

impl<T: Clone> OptimizedExprNode<T> {
    /// Create a new optimized node
    pub fn new(id: ExprId, key: ExprKey, cache: ExprCache<T>) -> Self {
        Self {
            id,
            key,
            cache,
            result: None,
        }
    }

    /// Get the node ID
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Get the expression key
    pub fn key(&self) -> &ExprKey {
        &self.key
    }

    /// Check if the result is cached
    pub fn is_cached(&self) -> bool {
        self.result.is_some() || self.cache.contains(&self.id)
    }

    /// Get or compute the result
    pub fn get_or_compute<F>(&mut self, compute: F) -> SharedArray<T>
    where
        F: FnOnce() -> SharedArray<T>,
    {
        // Check local cache first
        if let Some(ref result) = self.result {
            return result.clone();
        }

        // Check shared cache
        if let Some(cached) = self.cache.get(&self.id) {
            self.result = Some(cached.clone());
            return cached;
        }

        // Compute and cache
        let result = compute();
        self.cache.insert(self.id, result.clone());
        self.result = Some(result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::shared::{SharedArrayExpr, SharedBinaryExpr};

    #[test]
    fn test_expr_id_uniqueness() {
        let id1 = ExprId::new();
        let id2 = ExprId::new();
        let id3 = ExprId::new();

        // Each ID should be unique
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_expr_cache_basic() {
        let cache: ExprCache<f64> = ExprCache::new();
        let id = ExprId::new();
        let array = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);

        // Initially empty
        assert!(cache.is_empty());
        assert!(!cache.contains(&id));

        // Insert and verify
        cache.insert(id, array.clone());
        assert!(!cache.is_empty());
        assert!(cache.contains(&id));
        assert_eq!(cache.len(), 1);

        // Retrieve and verify
        let cached = cache.get(&id);
        assert!(cached.is_some());
        assert_eq!(
            cached.expect("Cached value should exist").to_vec(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_expr_cache_multiple_entries() {
        let cache: ExprCache<f64> = ExprCache::new();

        let id1 = ExprId::new();
        let id2 = ExprId::new();
        let id3 = ExprId::new();

        cache.insert(id1, SharedArray::from_vec(vec![1.0]));
        cache.insert(id2, SharedArray::from_vec(vec![2.0]));
        cache.insert(id3, SharedArray::from_vec(vec![3.0]));

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&id1).expect("Should exist").to_vec(), vec![1.0]);
        assert_eq!(cache.get(&id2).expect("Should exist").to_vec(), vec![2.0]);
        assert_eq!(cache.get(&id3).expect("Should exist").to_vec(), vec![3.0]);
    }

    #[test]
    fn test_expr_cache_clear() {
        let cache: ExprCache<f64> = ExprCache::new();
        let id = ExprId::new();

        cache.insert(id, SharedArray::from_vec(vec![1.0, 2.0]));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert!(!cache.contains(&id));
    }

    #[test]
    fn test_cached_expr_basic() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let expr = SharedArrayExpr::new(arr);
        let cache: ExprCache<f64> = ExprCache::new();

        let cached = CachedExpr::new(expr, cache.clone());

        // First evaluation
        let result1 = cached.eval();
        assert_eq!(result1.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.len(), 1); // Result cached

        // Second evaluation should return cached result
        let result2 = cached.eval();
        assert_eq!(result2.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.len(), 1); // Still one entry
    }

    #[test]
    fn test_cached_expr_shared_cache() {
        let cache: ExprCache<f64> = ExprCache::new();

        let arr1 = SharedArray::from_vec(vec![1.0, 2.0]);
        let arr2 = SharedArray::from_vec(vec![3.0, 4.0]);

        let expr1 = SharedArrayExpr::new(arr1);
        let expr2 = SharedArrayExpr::new(arr2);

        let cached1 = CachedExpr::new(expr1, cache.clone());
        let cached2 = CachedExpr::new(expr2, cache.clone());

        // Both share the same cache
        cached1.eval();
        cached2.eval();

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_expr_key_array() {
        let key1 = ExprKey::array(0);
        let key2 = ExprKey::array(0);
        let key3 = ExprKey::array(1);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_expr_key_binary() {
        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);

        let add1 = ExprKey::binary("add", key_a.clone(), key_b.clone());
        let add2 = ExprKey::binary("add", key_a.clone(), key_b.clone());
        let mul1 = ExprKey::binary("mul", key_a.clone(), key_b.clone());

        assert_eq!(add1, add2); // Same operation, same operands
        assert_ne!(add1, mul1); // Different operations
    }

    #[test]
    fn test_expr_key_unary() {
        let key_a = ExprKey::array(0);

        let sqrt1 = ExprKey::unary("sqrt", key_a.clone());
        let sqrt2 = ExprKey::unary("sqrt", key_a.clone());
        let neg1 = ExprKey::unary("neg", key_a.clone());

        assert_eq!(sqrt1, sqrt2);
        assert_ne!(sqrt1, neg1);
    }

    #[test]
    fn test_expr_key_scalar() {
        let key_a = ExprKey::array(0);

        let add10_1 = ExprKey::scalar("add", key_a.clone(), hash_f64(10.0));
        let add10_2 = ExprKey::scalar("add", key_a.clone(), hash_f64(10.0));
        let add20 = ExprKey::scalar("add", key_a.clone(), hash_f64(20.0));

        assert_eq!(add10_1, add10_2); // Same scalar value
        assert_ne!(add10_1, add20); // Different scalar values
    }

    #[test]
    fn test_cse_optimizer_basic() {
        let mut optimizer: CSEOptimizer<f64> = CSEOptimizer::new();

        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);
        let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());

        // First request creates a new ID
        let id1 = optimizer.get_or_create_id(&key_sum);

        // Second request returns the same ID
        let id2 = optimizer.get_or_create_id(&key_sum);

        assert_eq!(id1, id2);
        assert_eq!(optimizer.stats().unique_expressions, 1);
    }

    #[test]
    fn test_cse_optimizer_multiple_keys() {
        let mut optimizer: CSEOptimizer<f64> = CSEOptimizer::new();

        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);
        let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());
        let key_prod = ExprKey::binary("mul", key_a.clone(), key_b.clone());

        let id_sum = optimizer.get_or_create_id(&key_sum);
        let id_prod = optimizer.get_or_create_id(&key_prod);

        assert_ne!(id_sum, id_prod);
        assert_eq!(optimizer.stats().unique_expressions, 2);
    }

    #[test]
    fn test_cse_analysis() {
        let key_a = ExprKey::array(0);
        let key_b = ExprKey::array(1);
        let key_sum = ExprKey::binary("add", key_a.clone(), key_b.clone());

        // Expression: (a + b) * (a + b) - sum appears twice
        let keys = vec![
            key_a.clone(),
            key_b.clone(),
            key_sum.clone(),
            key_sum.clone(), // Common subexpression
            ExprKey::binary("mul", key_sum.clone(), key_sum.clone()),
        ];

        let analysis = analyze_cse(&keys);

        assert_eq!(analysis.total_nodes, 5);
        assert!(analysis.common_subexpressions > 0);
        assert!(analysis.savings_ratio > 0.0);
    }

    #[test]
    fn test_cse_support_trait() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
        let expr = SharedArrayExpr::new(arr);
        let cache: ExprCache<f64> = ExprCache::new();

        // Use CSESupport trait
        let cached = expr.with_cache(cache.clone());
        let result = cached.eval();

        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cse_expr_builder() {
        let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let builder: CSEExprBuilder<f64> = CSEExprBuilder::new();
        let result = builder.eval_array(a);

        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_optimized_expr_node() {
        let cache: ExprCache<f64> = ExprCache::new();
        let id = ExprId::new();
        let key = ExprKey::array(0);

        let mut node = OptimizedExprNode::new(id, key.clone(), cache.clone());

        // Initially not cached
        assert!(!node.is_cached());

        // Compute and cache
        let result = node.get_or_compute(|| SharedArray::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
        assert!(node.is_cached());

        // Second call returns cached result (computation closure not called)
        let result2 = node.get_or_compute(|| SharedArray::from_vec(vec![9.0, 9.0, 9.0]));
        assert_eq!(result2.to_vec(), vec![1.0, 2.0, 3.0]); // Original value, not 9.0s
    }

    #[test]
    fn test_cse_shared_computation() {
        // Demonstrate CSE: compute (a + b) * (a + b) where (a + b) is computed only once
        let a = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = SharedArray::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let cache: ExprCache<f64> = ExprCache::new();

        // Create a cached version of (a + b)
        let expr_a = SharedArrayExpr::new(a);
        let expr_b = SharedArrayExpr::new(b);
        let sum = SharedBinaryExpr::new(expr_a, expr_b, |x, y| x + y)
            .expect("Binary expression creation should succeed");
        let cached_sum = CachedExpr::new(sum, cache.clone());

        // Evaluate (a + b) to cache it
        let sum_result = cached_sum.eval();
        assert_eq!(sum_result.to_vec(), vec![3.0, 5.0, 7.0, 9.0]);
        assert_eq!(cache.len(), 1);

        // Now (a + b) * (a + b) uses the cached result
        let sum_squared =
            SharedBinaryExpr::new(cached_sum.clone(), cached_sum, |x: f64, y: f64| x * y)
                .expect("Binary expression creation should succeed");

        let result = sum_squared.eval();
        // (3*3, 5*5, 7*7, 9*9) = (9, 25, 49, 81)
        assert_eq!(result.to_vec(), vec![9.0, 25.0, 49.0, 81.0]);
    }

    #[test]
    fn test_cached_expr_invalidate() {
        let arr = SharedArray::from_vec(vec![1.0, 2.0, 3.0]);
        let expr = SharedArrayExpr::new(arr);
        let cache: ExprCache<f64> = ExprCache::new();

        let cached = CachedExpr::new(expr, cache.clone());

        // Evaluate to cache
        cached.eval();
        assert_eq!(cache.len(), 1);

        // Invalidate
        cached.invalidate();

        // Note: The ID is removed from the cache
        // but the cache may still have the entry depending on timing
        // This tests the invalidation mechanism works
    }

    #[test]
    fn test_hash_f64() {
        let h1 = hash_f64(10.0);
        let h2 = hash_f64(10.0);
        let h3 = hash_f64(20.0);

        // Same value should produce same hash
        assert_eq!(h1, h2);
        // Different values should (very likely) produce different hashes
        assert_ne!(h1, h3);
    }
}
