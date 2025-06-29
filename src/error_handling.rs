//! Error handling configuration for NumRS2
//!
//! This module provides error handling configuration similar to NumPy's seterr, geterr, and errstate.
//! It allows controlling how floating-point errors (like division by zero, overflow, underflow, etc.)
//! are handled in NumRS2 operations.

use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};

/// Error handling behavior for different types of floating-point errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorAction {
    /// Ignore the error (continue execution)
    Ignore,
    /// Issue a warning but continue execution
    #[default]
    Warn,
    /// Raise an exception/error
    Raise,
    /// Call a user-defined callback function
    Call,
}

impl std::fmt::Display for ErrorAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorAction::Ignore => write!(f, "ignore"),
            ErrorAction::Warn => write!(f, "warn"),
            ErrorAction::Raise => write!(f, "raise"),
            ErrorAction::Call => write!(f, "call"),
        }
    }
}

impl std::str::FromStr for ErrorAction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ignore" => Ok(ErrorAction::Ignore),
            "warn" => Ok(ErrorAction::Warn),
            "raise" => Ok(ErrorAction::Raise),
            "call" => Ok(ErrorAction::Call),
            _ => Err(format!("Invalid error action: {}", s)),
        }
    }
}

/// Configuration for error handling
#[derive(Debug, Clone)]
pub struct ErrorState {
    /// How to handle division by zero
    pub divide: ErrorAction,
    /// How to handle overflow
    pub over: ErrorAction,
    /// How to handle underflow
    pub under: ErrorAction,
    /// How to handle invalid operations (like sqrt of negative number)
    pub invalid: ErrorAction,
}

impl Default for ErrorState {
    fn default() -> Self {
        Self {
            divide: ErrorAction::Warn,
            over: ErrorAction::Warn,
            under: ErrorAction::Ignore,
            invalid: ErrorAction::Warn,
        }
    }
}

impl ErrorState {
    /// Create a new error state with all actions set to the same value
    pub fn new(action: ErrorAction) -> Self {
        Self {
            divide: action,
            over: action,
            under: action,
            invalid: action,
        }
    }

    /// Create a new error state with specific actions for each error type
    pub fn with_actions(
        divide: ErrorAction,
        over: ErrorAction,
        under: ErrorAction,
        invalid: ErrorAction,
    ) -> Self {
        Self {
            divide,
            over,
            under,
            invalid,
        }
    }
}

// Global error state for NumRS2
lazy_static! {
    static ref GLOBAL_ERROR_STATE: Arc<Mutex<ErrorState>> =
        Arc::new(Mutex::new(ErrorState::default()));
}

/// User-defined error callback function type
pub type ErrorCallback = Arc<dyn Fn(&str) + Send + Sync>;

// Global error callback
lazy_static! {
    static ref GLOBAL_ERROR_CALLBACK: Arc<Mutex<Option<ErrorCallback>>> =
        Arc::new(Mutex::new(None));
}

/// Set the error handling behavior for floating-point errors
///
/// # Arguments
///
/// * `all` - Action to take for all error types (if specified, overrides individual settings)
/// * `divide` - Action for division by zero
/// * `over` - Action for overflow
/// * `under` - Action for underflow
/// * `invalid` - Action for invalid operations
///
/// # Returns
///
/// The previous error state
///
/// # Examples
///
/// ```
/// use numrs2::error_handling::{seterr, ErrorAction};
///
/// // Set all errors to raise exceptions
/// let old_state = seterr(Some(ErrorAction::Raise), None, None, None, None);
///
/// // Set specific error handling
/// let old_state = seterr(
///     None,
///     Some(ErrorAction::Raise),  // Division by zero raises
///     Some(ErrorAction::Warn),   // Overflow warns
///     Some(ErrorAction::Ignore), // Underflow ignored
///     Some(ErrorAction::Warn),   // Invalid warns
/// );
/// ```
pub fn seterr(
    all: Option<ErrorAction>,
    divide: Option<ErrorAction>,
    over: Option<ErrorAction>,
    under: Option<ErrorAction>,
    invalid: Option<ErrorAction>,
) -> ErrorState {
    let mut state = GLOBAL_ERROR_STATE.lock().unwrap();
    let old_state = state.clone();

    if let Some(action) = all {
        *state = ErrorState::new(action);
    }

    if let Some(action) = divide {
        state.divide = action;
    }
    if let Some(action) = over {
        state.over = action;
    }
    if let Some(action) = under {
        state.under = action;
    }
    if let Some(action) = invalid {
        state.invalid = action;
    }

    old_state
}

/// Get the current error handling behavior
///
/// # Returns
///
/// The current error state
///
/// # Examples
///
/// ```
/// use numrs2::error_handling::geterr;
///
/// let current_state = geterr();
/// println!("Division by zero: {}", current_state.divide);
/// println!("Overflow: {}", current_state.over);
/// println!("Underflow: {}", current_state.under);
/// println!("Invalid: {}", current_state.invalid);
/// ```
pub fn geterr() -> ErrorState {
    GLOBAL_ERROR_STATE.lock().unwrap().clone()
}

/// Set the callback function for error handling
///
/// # Arguments
///
/// * `callback` - Function to call when errors occur (if ErrorAction::Call is set)
///
/// # Examples
///
/// ```
/// use numrs2::error_handling::{seterrcall, ErrorAction, seterr};
/// use std::sync::Arc;
///
/// // Set a custom error callback
/// seterrcall(Some(Arc::new(|msg: &str| {
///     eprintln!("NumRS2 Error: {}", msg);
/// })));
///
/// // Configure to use the callback for division by zero
/// seterr(None, Some(ErrorAction::Call), None, None, None);
/// ```
pub fn seterrcall(callback: Option<ErrorCallback>) {
    let mut cb = GLOBAL_ERROR_CALLBACK.lock().unwrap();
    *cb = callback;
}

/// Get the current error callback
///
/// # Returns
///
/// The current error callback (if any)
pub fn geterrcall() -> Option<ErrorCallback> {
    GLOBAL_ERROR_CALLBACK.lock().unwrap().clone()
}

/// Error types that can occur in floating-point operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatingPointError {
    /// Division by zero
    DivideByZero,
    /// Overflow
    Overflow,
    /// Underflow
    Underflow,
    /// Invalid operation
    Invalid,
}

impl std::fmt::Display for FloatingPointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FloatingPointError::DivideByZero => write!(f, "divide by zero"),
            FloatingPointError::Overflow => write!(f, "overflow"),
            FloatingPointError::Underflow => write!(f, "underflow"),
            FloatingPointError::Invalid => write!(f, "invalid operation"),
        }
    }
}

/// Handle a floating-point error according to the current error state
///
/// # Arguments
///
/// * `error_type` - The type of error that occurred
/// * `message` - Descriptive message about the error
///
/// # Returns
///
/// `true` if the operation should continue, `false` if it should abort
pub fn handle_error(error_type: FloatingPointError, message: &str) -> bool {
    let state = geterr();
    let action = match error_type {
        FloatingPointError::DivideByZero => state.divide,
        FloatingPointError::Overflow => state.over,
        FloatingPointError::Underflow => state.under,
        FloatingPointError::Invalid => state.invalid,
    };

    match action {
        ErrorAction::Ignore => true,
        ErrorAction::Warn => {
            eprintln!("Warning: {} - {}", error_type, message);
            true
        }
        ErrorAction::Raise => {
            panic!("NumRS2 Error: {} - {}", error_type, message);
        }
        ErrorAction::Call => {
            if let Some(callback) = geterrcall() {
                callback(&format!("{} - {}", error_type, message));
            }
            true
        }
    }
}

/// Context manager for temporarily changing error handling behavior
///
/// Similar to NumPy's errstate context manager. This allows you to temporarily
/// change the error handling behavior and automatically restore it when done.
///
/// # Examples
///
/// ```
/// use numrs2::error_handling::{errstate, ErrorAction};
///
/// {
///     let _guard = errstate()
///         .divide(ErrorAction::Ignore)
///         .over(ErrorAction::Raise)
///         .enter();
///     
///     // Operations here will ignore division by zero and raise on overflow
///     // Error state is automatically restored when _guard goes out of scope
/// }
/// ```
pub struct ErrorStateGuard {
    old_state: ErrorState,
}

impl Drop for ErrorStateGuard {
    fn drop(&mut self) {
        // Restore the old error state
        let mut state = GLOBAL_ERROR_STATE.lock().unwrap();
        *state = self.old_state.clone();
    }
}

/// Builder for creating temporary error state contexts
pub struct ErrorStateBuilder {
    divide: Option<ErrorAction>,
    over: Option<ErrorAction>,
    under: Option<ErrorAction>,
    invalid: Option<ErrorAction>,
}

impl ErrorStateBuilder {
    /// Set the action for division by zero errors
    pub fn divide(mut self, action: ErrorAction) -> Self {
        self.divide = Some(action);
        self
    }

    /// Set the action for overflow errors
    pub fn over(mut self, action: ErrorAction) -> Self {
        self.over = Some(action);
        self
    }

    /// Set the action for underflow errors
    pub fn under(mut self, action: ErrorAction) -> Self {
        self.under = Some(action);
        self
    }

    /// Set the action for invalid operation errors
    pub fn invalid(mut self, action: ErrorAction) -> Self {
        self.invalid = Some(action);
        self
    }

    /// Enter the error state context
    ///
    /// # Returns
    ///
    /// A guard that will restore the previous error state when dropped
    pub fn enter(self) -> ErrorStateGuard {
        let old_state = seterr(None, self.divide, self.over, self.under, self.invalid);
        ErrorStateGuard { old_state }
    }
}

/// Create a new error state context manager
///
/// # Returns
///
/// A builder for configuring the temporary error state
///
/// # Examples
///
/// ```
/// use numrs2::error_handling::{errstate, ErrorAction};
///
/// {
///     let _guard = errstate()
///         .divide(ErrorAction::Ignore)
///         .over(ErrorAction::Raise)
///         .enter();
///     
///     // Your code here with modified error handling
/// } // Error state automatically restored here
/// ```
pub fn errstate() -> ErrorStateBuilder {
    ErrorStateBuilder {
        divide: None,
        over: None,
        under: None,
        invalid: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_state_default() {
        let state = ErrorState::default();
        assert_eq!(state.divide, ErrorAction::Warn);
        assert_eq!(state.over, ErrorAction::Warn);
        assert_eq!(state.under, ErrorAction::Ignore);
        assert_eq!(state.invalid, ErrorAction::Warn);
    }

    #[test]
    fn test_seterr_geterr() {
        // Save current state
        let original_state = geterr();

        // Set new state
        let _old_state = seterr(Some(ErrorAction::Raise), None, None, None, None);

        // Verify the new state
        let current_state = geterr();
        assert_eq!(current_state.divide, ErrorAction::Raise);
        assert_eq!(current_state.over, ErrorAction::Raise);
        assert_eq!(current_state.under, ErrorAction::Raise);
        assert_eq!(current_state.invalid, ErrorAction::Raise);

        // Restore original state
        seterr(
            None,
            Some(original_state.divide),
            Some(original_state.over),
            Some(original_state.under),
            Some(original_state.invalid),
        );
    }

    #[test]
    fn test_errstate_context_manager() {
        let original_state = geterr();

        {
            let _guard = errstate()
                .divide(ErrorAction::Ignore)
                .over(ErrorAction::Raise)
                .enter();

            let current_state = geterr();
            assert_eq!(current_state.divide, ErrorAction::Ignore);
            assert_eq!(current_state.over, ErrorAction::Raise);
            // Other values should remain unchanged
            assert_eq!(current_state.under, original_state.under);
            assert_eq!(current_state.invalid, original_state.invalid);
        }

        // State should be restored
        let restored_state = geterr();
        assert_eq!(restored_state.divide, original_state.divide);
        assert_eq!(restored_state.over, original_state.over);
        assert_eq!(restored_state.under, original_state.under);
        assert_eq!(restored_state.invalid, original_state.invalid);
    }

    #[test]
    fn test_error_action_from_str() {
        assert_eq!(
            "ignore".parse::<ErrorAction>().unwrap(),
            ErrorAction::Ignore
        );
        assert_eq!("warn".parse::<ErrorAction>().unwrap(), ErrorAction::Warn);
        assert_eq!("raise".parse::<ErrorAction>().unwrap(), ErrorAction::Raise);
        assert_eq!("call".parse::<ErrorAction>().unwrap(), ErrorAction::Call);

        assert!("invalid".parse::<ErrorAction>().is_err());
    }

    #[test]
    fn test_handle_error_ignore() {
        let _guard = errstate().divide(ErrorAction::Ignore).enter();
        let result = handle_error(FloatingPointError::DivideByZero, "test error");
        assert!(result); // Should continue
    }

    #[test]
    fn test_handle_error_warn() {
        let _guard = errstate().divide(ErrorAction::Warn).enter();
        let result = handle_error(FloatingPointError::DivideByZero, "test error");
        assert!(result); // Should continue but warn
    }

    #[test]
    #[should_panic(expected = "NumRS2 Error: divide by zero - test error")]
    fn test_handle_error_raise() {
        let _guard = errstate().divide(ErrorAction::Raise).enter();
        handle_error(FloatingPointError::DivideByZero, "test error");
    }
}
