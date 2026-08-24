//! Extended Modules for NumRS2
//!
//! This module contains advanced functionality that extends NumRS2's core capabilities.
//! All modules follow strict SCIRS2 integration policy and maintain production-grade
//! quality standards.
//!
//! # Module Organization
//!
//! ## Signal Processing and Analysis
//! - [`fft`]: Fast Fourier Transform operations
//! - [`fft_enhanced`]: Enhanced FFT with advanced windowing and convolution
//! - [`signal_processing`]: Signal filtering, windowing, and spectral analysis
//! - [`frequency_analysis`]: Frequency domain analysis tools
//! - [`spectral_analysis`]: Power spectral density and spectrograms
//! - [`wavelets`]: Wavelet transforms (DWT, CWT), multiresolution analysis, denoising
//!
//! ## Linear Algebra Extensions
//! - [`eigenvalues`]: Eigenvalue and eigenvector computations
//! - [`matrix_decomp`]: Matrix decompositions (LU, QR, Cholesky, Schur, SVD)
//! - [`sparse`]: Sparse matrix data structures and operations
//!
//! ## Mathematical Functions
//! - [`special`]: Special mathematical functions (Bessel, Gamma, Error functions, etc.)
//! - [`polynomial`]: Polynomial operations, fitting, and root finding
//!
//! ## Neural Networks and Machine Learning
//! - [`nn`]: Advanced neural network architectures (Transformers, Graph Neural Networks)
//! - [`rl`]: Reinforcement learning primitives (agents, environments, replay buffers)
//! - [`model_io`]: Model serialization, deployment formats, and checkpointing
//! - [`serving`]: Production ML serving (inference, model registry, preprocessing, monitoring)
//!
//! ## Time Series and Probabilistic Models
//! - [`timeseries`]: Time series analysis and forecasting
//! - [`probabilistic`]: Probabilistic models and Bayesian inference
//!
//! ## Quantum Computing
//! - [`quantum`]: Quantum computing simulation (state vectors, gates, circuits, algorithms)
//!
//! ## Computer Vision
//! - [`cv`]: Image processing, filtering, morphology, feature detection, geometric transforms
//!
//! ## Computational Geometry
//! - [`geometry`]: Convex hulls, Delaunay triangulation, Voronoi diagrams, polygon operations
//!
//! ## Finite Element Method
//! - [`fem`]: FEM solver for PDEs (heat conduction, linear elasticity, mesh generation)
//!
//! ## Graph Algorithms
//! - [`graph`]: Graph data structures and algorithms (traversal, shortest paths, MST, flow)
//!
//! ## Control Systems
//! - [`control`]: Transfer functions, state-space, stability analysis, PID controllers
//!
//! ## Information Theory
//! - [`information_theory`]: Entropy measures, divergence metrics, mutual information, coding theory
//!
//! ## Physical Constants
//! - [`constants`]: NIST/CODATA 2022 physical constants (fundamental, atomic, electromagnetic, units)
//!
//! # SCIRS2 Integration Policy
//!
//! All modules in `new_modules` strictly follow NumRS2's SCIRS2 integration policy:
//! - **Array Operations**: Use `scirs2_core::ndarray` (NEVER direct ndarray)
//! - **Random Numbers**: Use `scirs2_core::random` (NEVER direct rand)
//! - **Parallelization**: Use `scirs2_core::parallel_ops` (NEVER direct rayon)
//! - **Linear Algebra**: Use `scirs2_linalg` for matrix operations
//! - **Error Handling**: Base errors on `scirs2_core::error::CoreError`
//! - **Pure Rust**: 100% Pure Rust via SciRS2 ecosystem (no C/C++ dependencies)
//!
//! # Quality Standards
//!
//! All code maintains strict quality standards:
//! - No `unwrap()` calls in production code
//! - Comprehensive error handling with `Result<T, NumRs2Error>`
//! - Full documentation with mathematical formulas and citations
//! - Extensive test coverage (unit, integration, property-based)
//! - SIMD optimization where applicable
//! - Numerical stability guarantees

// Signal processing modules
//
// `fft` was merged into `crate::fft` (the former near-duplicate
// `src/new_modules/fft.rs` implementation now lives there instead); this
// re-export keeps `crate::new_modules::fft::*` resolving to the same
// items for the handful of sibling modules in this directory that still
// spell the path that way, without maintaining two copies of the code.
pub use crate::fft;
pub mod fft_enhanced;
pub mod frequency_analysis;
pub mod signal_processing;
pub mod spectral_analysis;
pub mod wavelets;

// Linear algebra extensions
pub mod eigenvalues;
pub mod matrix_decomp;
pub mod sparse;

// Mathematical functions
pub mod polynomial;
pub mod special;

// Neural networks and machine learning
pub mod model_io;
pub mod nn;
pub mod rl;
pub mod serving;

// Time series and probabilistic models
pub mod probabilistic;
pub mod timeseries;

// Quantum computing
pub mod quantum;

// Computer Vision
pub mod cv;

// Computational Geometry
pub mod geometry;

// Finite Element Method
pub mod fem;

// Graph Algorithms
pub mod graph;

// Control Systems
pub mod control;

// Information Theory
pub mod information_theory;

// Physical Constants
pub mod constants;

// Survival Analysis
pub mod survival;

// Causal Inference
pub mod causal;

// Bioinformatics
pub mod bioinformatics;

// Combinatorics
pub mod combinatorics;

// String Algorithms
pub mod string_algorithms;

// Streaming/Online Statistics
pub mod streaming_stats;
