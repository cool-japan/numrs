//! Advanced Exponential Smoothing Methods for Time Series Analysis
//!
//! This module implements the ETS (Error, Trend, Seasonal) framework for
//! exponential smoothing, providing a unified interface for:
//!
//! - **Simple Exponential Smoothing (SES)**: Level-only smoothing (ETS(A,N,N))
//! - **Holt's Linear Method**: Level + additive trend (ETS(A,A,N))
//! - **Damped Trend Method**: Level + damped additive trend (ETS(A,Ad,N))
//! - **Holt-Winters Additive**: Level + trend + additive seasonality (ETS(A,A,A))
//! - **Holt-Winters Multiplicative**: Level + trend + multiplicative seasonality (ETS(A,A,M))
//! - **Damped Holt-Winters**: Damped trend variants of seasonal models
//!
//! ## Features
//!
//! - Parameter optimization via grid search minimizing MSE
//! - Prediction intervals using analytical variance formulas
//! - AIC/BIC information criteria for model selection
//! - Robust component initialization
//!
//! ## References
//!
//! - Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
//! - Hyndman, R.J., Koehler, A.B., Ord, J.K., & Snyder, R.D. (2008).
//!   *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
//! - Gardner, E.S. Jr (2006). "Exponential smoothing: The state of the art - Part II."
//!   *International Journal of Forecasting*, 22(4), 637-666.

pub mod core;
pub mod helpers;
pub mod optimization;
pub mod types;

mod tests;

pub use core::ExponentialSmoothing;
pub use optimization::{optimize_parameters, select_best_model};
pub use types::{
    ExponentialSmoothingForecast, ExponentialSmoothingResult, InformationCriteria,
    OptimizationConfig, SeasonalComponent, TrendComponent,
};
