//! Expression Templates for Lazy Evaluation
//!
//! This module implements expression templates to enable lazy evaluation and
//! operation fusion. Instead of computing intermediate results immediately,
//! operations build an expression tree that can be optimized and evaluated
//! efficiently.
//!
//! # Benefits
//!
//! - **Eliminates intermediate allocations**: `(a + b) * c` computed in one pass
//! - **Kernel fusion**: Multiple operations fused into single SIMD loop
//! - **Optimized memory access**: Better cache utilization
//! - **Deferred evaluation**: Computation only when result is needed
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::prelude::*;
//! use numrs2::expr::*;
//!
//! let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
//! let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
//!
//! // Create expression manually (operator overloading has lifetime issues)
//! let expr = BinaryExpr::new(
//!     ArrayExpr::new(&a),
//!     ArrayExpr::new(&b),
//!     |x, y| x + y
//! ).expect("creating binary expression should succeed");
//!
//! // Evaluation happens here
//! let result = expr.eval();
//! ```
//!
//! # Current Status
//!
//! This module provides the foundational infrastructure for expression templates.
//! Operator overloading has Rust lifetime challenges that need further work.
//! The core trait and types are functional and serve as a basis for future optimization.
//!
//! # Module Structure
//!
//! - [`core`]: Core traits (`Expr`, `LazyEval`) and basic expression types
//! - [`enhanced`]: Enhanced expression types (reduction, conditional, clip, broadcast)
//! - [`simd_eval`]: SIMD-optimized batch evaluation
//! - [`builder`]: Fluent API for building expressions
//! - [`shared`]: Reference-counted expression types (no lifetime constraints)
//! - [`cse`]: Common Subexpression Elimination (CSE) optimization
//! - [`owned`]: **Owned, lifetime-free expression templates that actually
//!   fuse** ([`ExprNode`] / [`IntoExpr`]) -- start here. Its evaluator lives
//!   in the crate-internal `fused_eval` module.
//!
//! # Which expression API to use
//!
//! [`owned`] is the one that fuses. Its [`ExprNode`] owns its operands (an
//! O(1) copy-on-write clone per leaf), so it carries no lifetime parameter,
//! supports ordinary operator syntax, and evaluates the common tree shapes in
//! one pass over flat slices (wider trees fall back to ordinary eager
//! evaluation, at eager's speed -- see [`owned`] for exactly which shapes
//! fuse):
//!
//! ```
//! use numrs2::prelude::*;
//!
//! let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
//! let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
//! let c = Array::from_vec(vec![2.0, 2.0, 2.0]);
//!
//! let fused = (a.expr() + b.expr() * c.expr()).eval()?;
//! assert_eq!(fused.to_vec(), vec![9.0, 12.0, 15.0]);
//! # Ok::<(), numrs2::error::NumRs2Error>(())
//! ```
//!
//! The older borrowing types below ([`core`], [`shared`], [`simd_eval`],
//! [`builder`], [`cse`], [`optimized_eval`], [`fusion`]) predate
//! copy-on-write storage and evaluate per element through a `get_flat`
//! dispatch chain; they remain for compatibility, but new code that wants
//! fusion should use [`owned`].

pub mod builder;
pub mod core;
pub mod cse;
pub mod enhanced;
pub(crate) mod fused_eval;
pub mod fusion;
pub mod optimized_eval;
pub mod owned;
pub mod shared;
pub mod simd_eval;

// Re-export the owned (fusing) expression templates
pub use owned::{BinOp, ExprNode, IntoExpr, UnaryOp};

// Re-export core types
pub use self::core::{ArrayExpr, BinaryExpr, Expr, LazyEval, ScalarExpr, UnaryExpr};

// Re-export enhanced types
pub use enhanced::{BroadcastScalarExpr, ClipExpr, ReductionExpr, WhereExpr};

// Re-export SIMD evaluation
pub use simd_eval::SimdEval;

// Re-export builder types and utility functions
pub use builder::{expr_prod, expr_sum, fma, ExprBuilder};

// Re-export shared expression types
pub use shared::{
    SharedArrayExpr, SharedBinaryExpr, SharedExpr, SharedExprBuilder, SharedScalarExpr,
    SharedUnaryExpr,
};

// Re-export CSE types
pub use cse::{
    analyze_cse, hash_f64, CSEAnalysisResult, CSEExprBuilder, CSEOptimizer, CSEStats, CSESupport,
    CachedExpr, ExprCache, ExprId, ExprKey, OptimizedExprNode,
};

// Re-export optimized evaluation types
pub use optimized_eval::{
    BinaryOpType, BufferPool, FusedBinaryScalarExpr, SimdBinaryEvaluator, SimdBinaryExpr,
    SimdExprEval,
};

// Re-export fusion types
pub use fusion::{
    fused_dot_product, fused_scalar_broadcast_f32, fused_scalar_broadcast_f64, fused_sum_abs_diff,
    fused_sum_of_squares, simd_fma_arrays, simd_fused_dot_product, simd_fused_scalar_broadcast,
    FusedElementWiseChain, FusedMultiplyAdd, FusedOp, FusedQuadOp, FusedReduction,
    FusedScalarBroadcast, FusedUnaryChain, FusionAnalysis, FusionBuilder, FusionDetector,
    FusionPattern,
};
