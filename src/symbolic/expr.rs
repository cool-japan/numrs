//! Expression tree representation for symbolic mathematics
//!
//! This module defines the core `Expr` type and its operations.

use crate::error::{NumRs2Error, Result};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Symbolic expression tree
///
/// Represents mathematical expressions as a tree structure that can be
/// manipulated symbolically before numerical evaluation.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::Expr;
///
/// // Create expression: (x + 1) * 2
/// let x = Expr::var("x");
/// let expr = (x + 1.0) * 2.0;
///
/// // Evaluate at x = 3
/// let mut vars = std::collections::HashMap::new();
/// vars.insert("x".to_string(), 3.0);
/// let result = expr.eval(&vars).expect("valid expression evaluation");
/// assert_eq!(result, 8.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Constant value
    Constant(f64),
    /// Variable with a name
    Variable(String),
    /// Addition: f + g
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: f - g
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: f * g
    Mul(Box<Expr>, Box<Expr>),
    /// Division: f / g
    Div(Box<Expr>, Box<Expr>),
    /// Power: f^g
    Pow(Box<Expr>, Box<Expr>),
    /// Negation: -f
    Neg(Box<Expr>),
    /// Sine function
    Sin(Box<Expr>),
    /// Cosine function
    Cos(Box<Expr>),
    /// Tangent function
    Tan(Box<Expr>),
    /// Exponential function
    Exp(Box<Expr>),
    /// Natural logarithm
    Ln(Box<Expr>),
    /// Square root
    Sqrt(Box<Expr>),
}

impl Expr {
    /// Create a constant expression
    pub fn constant(value: f64) -> Self {
        Expr::Constant(value)
    }

    /// Create a variable expression
    pub fn var(name: &str) -> Self {
        Expr::Variable(name.to_string())
    }

    /// Create a power expression: self^exponent
    pub fn pow(self, exponent: impl Into<Expr>) -> Self {
        Expr::Pow(Box::new(self), Box::new(exponent.into()))
    }

    /// Create a sine expression
    pub fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    /// Create a cosine expression
    pub fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    /// Create a tangent expression
    pub fn tan(self) -> Self {
        Expr::Tan(Box::new(self))
    }

    /// Create an exponential expression
    pub fn exp(self) -> Self {
        Expr::Exp(Box::new(self))
    }

    /// Create a natural logarithm expression
    pub fn ln(self) -> Self {
        Expr::Ln(Box::new(self))
    }

    /// Create a square root expression
    pub fn sqrt(self) -> Self {
        Expr::Sqrt(Box::new(self))
    }

    /// Check if the expression contains a specific variable
    pub fn contains_var(&self, name: &str) -> bool {
        match self {
            Expr::Constant(_) => false,
            Expr::Variable(v) => v == name,
            Expr::Add(f, g)
            | Expr::Sub(f, g)
            | Expr::Mul(f, g)
            | Expr::Div(f, g)
            | Expr::Pow(f, g) => f.contains_var(name) || g.contains_var(name),
            Expr::Neg(f)
            | Expr::Sin(f)
            | Expr::Cos(f)
            | Expr::Tan(f)
            | Expr::Exp(f)
            | Expr::Ln(f)
            | Expr::Sqrt(f) => f.contains_var(name),
        }
    }

    /// Substitute a variable with another expression
    pub fn substitute(&self, var: &str, expr: &Expr) -> Expr {
        match self {
            Expr::Constant(c) => Expr::Constant(*c),
            Expr::Variable(v) => {
                if v == var {
                    expr.clone()
                } else {
                    Expr::Variable(v.clone())
                }
            }
            Expr::Add(f, g) => Expr::Add(
                Box::new(f.substitute(var, expr)),
                Box::new(g.substitute(var, expr)),
            ),
            Expr::Sub(f, g) => Expr::Sub(
                Box::new(f.substitute(var, expr)),
                Box::new(g.substitute(var, expr)),
            ),
            Expr::Mul(f, g) => Expr::Mul(
                Box::new(f.substitute(var, expr)),
                Box::new(g.substitute(var, expr)),
            ),
            Expr::Div(f, g) => Expr::Div(
                Box::new(f.substitute(var, expr)),
                Box::new(g.substitute(var, expr)),
            ),
            Expr::Pow(f, g) => Expr::Pow(
                Box::new(f.substitute(var, expr)),
                Box::new(g.substitute(var, expr)),
            ),
            Expr::Neg(f) => Expr::Neg(Box::new(f.substitute(var, expr))),
            Expr::Sin(f) => Expr::Sin(Box::new(f.substitute(var, expr))),
            Expr::Cos(f) => Expr::Cos(Box::new(f.substitute(var, expr))),
            Expr::Tan(f) => Expr::Tan(Box::new(f.substitute(var, expr))),
            Expr::Exp(f) => Expr::Exp(Box::new(f.substitute(var, expr))),
            Expr::Ln(f) => Expr::Ln(Box::new(f.substitute(var, expr))),
            Expr::Sqrt(f) => Expr::Sqrt(Box::new(f.substitute(var, expr))),
        }
    }

    /// Evaluate the expression numerically with the given variable values
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Expr::Constant(c) => Ok(*c),
            Expr::Variable(v) => vars.get(v).copied().ok_or_else(|| {
                NumRs2Error::ValueError(format!("Variable '{}' not found in evaluation context", v))
            }),
            Expr::Add(f, g) => Ok(f.eval(vars)? + g.eval(vars)?),
            Expr::Sub(f, g) => Ok(f.eval(vars)? - g.eval(vars)?),
            Expr::Mul(f, g) => Ok(f.eval(vars)? * g.eval(vars)?),
            Expr::Div(f, g) => {
                let denom = g.eval(vars)?;
                if denom.abs() < f64::EPSILON {
                    Err(NumRs2Error::ValueError(
                        "Division by zero in symbolic expression evaluation".to_string(),
                    ))
                } else {
                    Ok(f.eval(vars)? / denom)
                }
            }
            Expr::Pow(f, g) => {
                let base = f.eval(vars)?;
                let exp = g.eval(vars)?;
                if base < 0.0 && exp.fract() != 0.0 {
                    Err(NumRs2Error::ValueError(
                        "Cannot raise negative number to non-integer power".to_string(),
                    ))
                } else {
                    Ok(base.powf(exp))
                }
            }
            Expr::Neg(f) => Ok(-f.eval(vars)?),
            Expr::Sin(f) => Ok(f.eval(vars)?.sin()),
            Expr::Cos(f) => Ok(f.eval(vars)?.cos()),
            Expr::Tan(f) => Ok(f.eval(vars)?.tan()),
            Expr::Exp(f) => Ok(f.eval(vars)?.exp()),
            Expr::Ln(f) => {
                let val = f.eval(vars)?;
                if val <= 0.0 {
                    Err(NumRs2Error::ValueError(
                        "Cannot take logarithm of non-positive number".to_string(),
                    ))
                } else {
                    Ok(val.ln())
                }
            }
            Expr::Sqrt(f) => {
                let val = f.eval(vars)?;
                if val < 0.0 {
                    Err(NumRs2Error::ValueError(
                        "Cannot take square root of negative number".to_string(),
                    ))
                } else {
                    Ok(val.sqrt())
                }
            }
        }
    }

    /// Convert expression to LaTeX format
    pub fn to_latex(&self) -> String {
        match self {
            Expr::Constant(c) => {
                if c.fract() == 0.0 && c.abs() < 1e10 {
                    format!("{}", *c as i64)
                } else {
                    format!("{}", c)
                }
            }
            Expr::Variable(v) => v.clone(),
            Expr::Add(f, g) => format!(
                "{} + {}",
                f.to_latex_with_parens(5),
                g.to_latex_with_parens(5)
            ),
            Expr::Sub(f, g) => format!(
                "{} - {}",
                f.to_latex_with_parens(5),
                g.to_latex_with_parens(6)
            ),
            Expr::Mul(f, g) => {
                let left = f.to_latex_with_parens(10);
                let right = g.to_latex_with_parens(10);
                format!("{} \\cdot {}", left, right)
            }
            Expr::Div(f, g) => format!("\\frac{{{}}}{{{}}}", f.to_latex(), g.to_latex()),
            Expr::Pow(f, g) => format!("{{{}}}^{{{}}}", f.to_latex_with_parens(20), g.to_latex()),
            Expr::Neg(f) => format!("-({})", f.to_latex()),
            Expr::Sin(f) => format!("\\sin({})", f.to_latex()),
            Expr::Cos(f) => format!("\\cos({})", f.to_latex()),
            Expr::Tan(f) => format!("\\tan({})", f.to_latex()),
            Expr::Exp(f) => format!("e^{{{}}}", f.to_latex()),
            Expr::Ln(f) => format!("\\ln({})", f.to_latex()),
            Expr::Sqrt(f) => format!("\\sqrt{{{}}}", f.to_latex()),
        }
    }

    /// Helper for LaTeX with parentheses based on precedence
    fn to_latex_with_parens(&self, parent_prec: u8) -> String {
        let self_prec = self.precedence();
        let latex = self.to_latex();
        if self_prec < parent_prec {
            format!("({})", latex)
        } else {
            latex
        }
    }

    /// Get operator precedence for proper parenthesization
    fn precedence(&self) -> u8 {
        match self {
            Expr::Constant(_) | Expr::Variable(_) => 100,
            Expr::Add(_, _) | Expr::Sub(_, _) => 5,
            Expr::Mul(_, _) | Expr::Div(_, _) => 10,
            Expr::Neg(_) => 15,
            Expr::Pow(_, _) => 20,
            Expr::Sin(_)
            | Expr::Cos(_)
            | Expr::Tan(_)
            | Expr::Exp(_)
            | Expr::Ln(_)
            | Expr::Sqrt(_) => 25,
        }
    }

    /// Convert expression to Python-compatible string
    pub fn to_python(&self) -> String {
        match self {
            Expr::Constant(c) => format!("{}", c),
            Expr::Variable(v) => v.clone(),
            Expr::Add(f, g) => format!("({} + {})", f.to_python(), g.to_python()),
            Expr::Sub(f, g) => format!("({} - {})", f.to_python(), g.to_python()),
            Expr::Mul(f, g) => format!("({} * {})", f.to_python(), g.to_python()),
            Expr::Div(f, g) => format!("({} / {})", f.to_python(), g.to_python()),
            Expr::Pow(f, g) => format!("({}**{})", f.to_python(), g.to_python()),
            Expr::Neg(f) => format!("-({})", f.to_python()),
            Expr::Sin(f) => format!("sin({})", f.to_python()),
            Expr::Cos(f) => format!("cos({})", f.to_python()),
            Expr::Tan(f) => format!("tan({})", f.to_python()),
            Expr::Exp(f) => format!("exp({})", f.to_python()),
            Expr::Ln(f) => format!("log({})", f.to_python()),
            Expr::Sqrt(f) => format!("sqrt({})", f.to_python()),
        }
    }
}

// Implement Display for human-readable output
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Constant(c) => write!(f, "{}", c),
            Expr::Variable(v) => write!(f, "{}", v),
            Expr::Add(left, right) => write!(f, "({} + {})", left, right),
            Expr::Sub(left, right) => write!(f, "({} - {})", left, right),
            Expr::Mul(left, right) => write!(f, "({} * {})", left, right),
            Expr::Div(left, right) => write!(f, "({} / {})", left, right),
            Expr::Pow(left, right) => write!(f, "({})^({})", left, right),
            Expr::Neg(expr) => write!(f, "-({})", expr),
            Expr::Sin(expr) => write!(f, "sin({})", expr),
            Expr::Cos(expr) => write!(f, "cos({})", expr),
            Expr::Tan(expr) => write!(f, "tan({})", expr),
            Expr::Exp(expr) => write!(f, "exp({})", expr),
            Expr::Ln(expr) => write!(f, "ln({})", expr),
            Expr::Sqrt(expr) => write!(f, "sqrt({})", expr),
        }
    }
}

// Operator overloading for convenient expression building

impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Expr {
        Expr::Div(Box::new(self), Box::new(rhs))
    }
}

impl Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

// Allow mixing Expr with f64
impl Add<f64> for Expr {
    type Output = Expr;

    fn add(self, rhs: f64) -> Expr {
        Expr::Add(Box::new(self), Box::new(Expr::Constant(rhs)))
    }
}

impl Sub<f64> for Expr {
    type Output = Expr;

    fn sub(self, rhs: f64) -> Expr {
        Expr::Sub(Box::new(self), Box::new(Expr::Constant(rhs)))
    }
}

impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Expr {
        Expr::Mul(Box::new(self), Box::new(Expr::Constant(rhs)))
    }
}

impl Div<f64> for Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Expr {
        Expr::Div(Box::new(self), Box::new(Expr::Constant(rhs)))
    }
}

// Allow f64 + Expr
impl Add<Expr> for f64 {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(Expr::Constant(self)), Box::new(rhs))
    }
}

impl Sub<Expr> for f64 {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Box::new(Expr::Constant(self)), Box::new(rhs))
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(Expr::Constant(self)), Box::new(rhs))
    }
}

impl Div<Expr> for f64 {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Expr {
        Expr::Div(Box::new(Expr::Constant(self)), Box::new(rhs))
    }
}

// Allow converting f64 to Expr
impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Expr::Constant(value)
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        Expr::Constant(value as f64)
    }
}

impl From<&str> for Expr {
    fn from(name: &str) -> Self {
        Expr::Variable(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_creation() {
        let x = Expr::var("x");
        let c = Expr::constant(5.0);

        assert!(matches!(x, Expr::Variable(_)));
        assert!(matches!(c, Expr::Constant(_)));
    }

    #[test]
    fn test_expr_operators() {
        let x = Expr::var("x");
        let y = Expr::var("y");

        let sum = x.clone() + y.clone();
        assert!(matches!(sum, Expr::Add(_, _)));

        let product = x.clone() * y.clone();
        assert!(matches!(product, Expr::Mul(_, _)));
    }

    #[test]
    fn test_expr_eval() {
        let x = Expr::var("x");
        let expr = x.clone() * x.clone() + x.clone() * 2.0 + 1.0;

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);

        let result = expr.eval(&vars);
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(16.0));
    }

    #[test]
    fn test_contains_var() {
        let x = Expr::var("x");
        let y = Expr::var("y");
        let expr = x.clone() * 2.0 + y.clone();

        assert!(expr.contains_var("x"));
        assert!(expr.contains_var("y"));
        assert!(!expr.contains_var("z"));
    }

    #[test]
    fn test_substitute() {
        let x = Expr::var("x");
        let expr = x.clone() * x.clone(); // x²

        let two = Expr::constant(2.0);
        let substituted = expr.substitute("x", &two); // 2²

        let vars = HashMap::new();
        let result = substituted.eval(&vars);
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(4.0));
    }

    #[test]
    fn test_trig_functions() {
        let x = Expr::var("x");
        let sin_expr = x.clone().sin();
        let cos_expr = x.clone().cos();

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);

        assert_eq!(sin_expr.eval(&vars).ok(), Some(0.0));
        assert_eq!(cos_expr.eval(&vars).ok(), Some(1.0));
    }

    #[test]
    fn test_division_by_zero_error() {
        let x = Expr::var("x");
        let expr = x / 0.0;

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);

        let result = expr.eval(&vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_latex_output() {
        let x = Expr::var("x");
        let expr = x.clone().pow(2.0) + x.clone() * 2.0 + 1.0;

        let latex = expr.to_latex();
        assert!(latex.contains("x"));
        assert!(latex.contains("^"));
    }
}
