//! Symbolic differentiation for mathematical expressions
//!
//! This module implements symbolic differentiation using the chain rule and
//! standard derivative rules for common mathematical functions.

use crate::error::{NumRs2Error, Result};
use crate::symbolic::expr::Expr;

/// Compute the symbolic derivative of an expression with respect to a variable
///
/// Uses the chain rule and standard differentiation rules to compute exact
/// symbolic derivatives.
///
/// # Arguments
///
/// * `expr` - The expression to differentiate
/// * `var` - The variable name to differentiate with respect to
///
/// # Returns
///
/// The symbolic derivative as a new expression
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let f = x.clone() * x.clone(); // f(x) = x²
/// let df = differentiate(&f, "x").expect("valid differentiation"); // f'(x) = 2x
/// ```
pub fn differentiate(expr: &Expr, var: &str) -> Result<Expr> {
    match expr {
        // d/dx(c) = 0
        Expr::Constant(_) => Ok(Expr::Constant(0.0)),

        // d/dx(x) = 1, d/dx(y) = 0
        Expr::Variable(v) => {
            if v == var {
                Ok(Expr::Constant(1.0))
            } else {
                Ok(Expr::Constant(0.0))
            }
        }

        // Sum rule: d/dx(f + g) = f' + g'
        Expr::Add(f, g) => {
            let df = differentiate(f, var)?;
            let dg = differentiate(g, var)?;
            Ok(Expr::Add(Box::new(df), Box::new(dg)))
        }

        // Difference rule: d/dx(f - g) = f' - g'
        Expr::Sub(f, g) => {
            let df = differentiate(f, var)?;
            let dg = differentiate(g, var)?;
            Ok(Expr::Sub(Box::new(df), Box::new(dg)))
        }

        // Product rule: d/dx(f * g) = f' * g + f * g'
        Expr::Mul(f, g) => {
            let df = differentiate(f, var)?;
            let dg = differentiate(g, var)?;
            Ok(Expr::Add(
                Box::new(Expr::Mul(Box::new(df), g.clone())),
                Box::new(Expr::Mul(f.clone(), Box::new(dg))),
            ))
        }

        // Quotient rule: d/dx(f / g) = (f' * g - f * g') / g²
        Expr::Div(f, g) => {
            let df = differentiate(f, var)?;
            let dg = differentiate(g, var)?;

            let numerator = Expr::Sub(
                Box::new(Expr::Mul(Box::new(df), g.clone())),
                Box::new(Expr::Mul(f.clone(), Box::new(dg))),
            );

            let denominator = Expr::Mul(g.clone(), g.clone());

            Ok(Expr::Div(Box::new(numerator), Box::new(denominator)))
        }

        // Power rule: d/dx(f^g) = g*f^(g-1)*f' + f^g*ln(f)*g'
        // Special case: if g is constant, use simplified power rule
        Expr::Pow(f, g) => {
            if !g.contains_var(var) {
                // Simplified power rule: d/dx(f^c) = c * f^(c-1) * f'
                let df = differentiate(f, var)?;
                let g_minus_1 = Expr::Sub(g.clone(), Box::new(Expr::Constant(1.0)));
                Ok(Expr::Mul(
                    Box::new(Expr::Mul(
                        g.clone(),
                        Box::new(Expr::Pow(f.clone(), Box::new(g_minus_1))),
                    )),
                    Box::new(df),
                ))
            } else if !f.contains_var(var) {
                // Exponential rule: d/dx(c^g) = c^g * ln(c) * g'
                let dg = differentiate(g, var)?;
                Ok(Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Pow(f.clone(), g.clone())),
                        Box::new(Expr::Ln(f.clone())),
                    )),
                    Box::new(dg),
                ))
            } else {
                // General power rule: d/dx(f^g) = g*f^(g-1)*f' + f^g*ln(f)*g'
                let df = differentiate(f, var)?;
                let dg = differentiate(g, var)?;

                let g_minus_1 = Expr::Sub(g.clone(), Box::new(Expr::Constant(1.0)));
                let term1 = Expr::Mul(
                    Box::new(Expr::Mul(
                        g.clone(),
                        Box::new(Expr::Pow(f.clone(), Box::new(g_minus_1))),
                    )),
                    Box::new(df),
                );

                let term2 = Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Pow(f.clone(), g.clone())),
                        Box::new(Expr::Ln(f.clone())),
                    )),
                    Box::new(dg),
                );

                Ok(Expr::Add(Box::new(term1), Box::new(term2)))
            }
        }

        // Negation rule: d/dx(-f) = -f'
        Expr::Neg(f) => {
            let df = differentiate(f, var)?;
            Ok(Expr::Neg(Box::new(df)))
        }

        // d/dx(sin(f)) = cos(f) * f'
        Expr::Sin(f) => {
            let df = differentiate(f, var)?;
            Ok(Expr::Mul(Box::new(Expr::Cos(f.clone())), Box::new(df)))
        }

        // d/dx(cos(f)) = -sin(f) * f'
        Expr::Cos(f) => {
            let df = differentiate(f, var)?;
            Ok(Expr::Mul(
                Box::new(Expr::Neg(Box::new(Expr::Sin(f.clone())))),
                Box::new(df),
            ))
        }

        // d/dx(tan(f)) = sec²(f) * f' = (1/cos²(f)) * f'
        Expr::Tan(f) => {
            let df = differentiate(f, var)?;
            let cos_f = Expr::Cos(f.clone());
            let sec_squared = Expr::Div(
                Box::new(Expr::Constant(1.0)),
                Box::new(Expr::Mul(Box::new(cos_f.clone()), Box::new(cos_f))),
            );
            Ok(Expr::Mul(Box::new(sec_squared), Box::new(df)))
        }

        // d/dx(exp(f)) = exp(f) * f'
        Expr::Exp(f) => {
            let df = differentiate(f, var)?;
            Ok(Expr::Mul(Box::new(Expr::Exp(f.clone())), Box::new(df)))
        }

        // d/dx(ln(f)) = f' / f
        Expr::Ln(f) => {
            let df = differentiate(f, var)?;
            Ok(Expr::Div(Box::new(df), f.clone()))
        }

        // d/dx(sqrt(f)) = f' / (2 * sqrt(f))
        Expr::Sqrt(f) => {
            let df = differentiate(f, var)?;
            let denominator = Expr::Mul(
                Box::new(Expr::Constant(2.0)),
                Box::new(Expr::Sqrt(f.clone())),
            );
            Ok(Expr::Div(Box::new(df), Box::new(denominator)))
        }
    }
}

/// Compute the gradient of an expression with respect to multiple variables
///
/// The gradient is a vector of partial derivatives.
///
/// # Arguments
///
/// * `expr` - The expression to differentiate
/// * `vars` - The variable names to differentiate with respect to
///
/// # Returns
///
/// A vector of symbolic derivatives, one for each variable
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let y = Expr::var("y");
/// let f = x.clone() * x.clone() + y.clone() * y.clone(); // f(x,y) = x² + y²
///
/// let grad = gradient(&f, &["x", "y"]).expect("valid gradient computation");
/// // grad = [2x, 2y]
/// ```
pub fn gradient(expr: &Expr, vars: &[&str]) -> Result<Vec<Expr>> {
    let mut grad = Vec::with_capacity(vars.len());
    for var in vars {
        grad.push(differentiate(expr, var)?);
    }
    Ok(grad)
}

/// Compute the Jacobian matrix for vector-valued functions
///
/// The Jacobian is a matrix of partial derivatives where J\[i\]\[j\] = ∂f_i/∂x_j
///
/// # Arguments
///
/// * `exprs` - Vector of expressions representing the output components
/// * `vars` - The variable names to differentiate with respect to
///
/// # Returns
///
/// A 2D vector representing the Jacobian matrix
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let y = Expr::var("y");
///
/// // f(x,y) = [x², xy, y²]
/// let exprs = vec![
///     x.clone() * x.clone(),
///     x.clone() * y.clone(),
///     y.clone() * y.clone(),
/// ];
///
/// let jac = jacobian(&exprs, &["x", "y"]).expect("valid jacobian computation");
/// // jac = [[2x, 0], [y, x], [0, 2y]]
/// ```
pub fn jacobian(exprs: &[Expr], vars: &[&str]) -> Result<Vec<Vec<Expr>>> {
    let mut jac = Vec::with_capacity(exprs.len());
    for expr in exprs {
        jac.push(gradient(expr, vars)?);
    }
    Ok(jac)
}

/// Compute the Hessian matrix (matrix of second derivatives)
///
/// The Hessian is a square matrix of second partial derivatives where H\[i\]\[j\] = ∂²f/∂x_i∂x_j
///
/// # Arguments
///
/// * `expr` - The expression to differentiate twice
/// * `vars` - The variable names to differentiate with respect to
///
/// # Returns
///
/// A 2D vector representing the Hessian matrix
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let y = Expr::var("y");
/// let f = x.clone() * x.clone() + x.clone() * y.clone() + y.clone() * y.clone();
///
/// let hess = hessian(&f, &["x", "y"]).expect("valid hessian computation");
/// // hess = [[2, 1], [1, 2]]
/// ```
pub fn hessian(expr: &Expr, vars: &[&str]) -> Result<Vec<Vec<Expr>>> {
    let mut hess = Vec::with_capacity(vars.len());
    for &var1 in vars {
        let mut row = Vec::with_capacity(vars.len());
        for &var2 in vars {
            let first_deriv = differentiate(expr, var1)?;
            let second_deriv = differentiate(&first_deriv, var2)?;
            row.push(second_deriv);
        }
        hess.push(row);
    }
    Ok(hess)
}

/// Compute the directional derivative of an expression
///
/// The directional derivative is the rate of change in a specific direction,
/// computed as ∇f · v where v is the direction vector.
///
/// # Arguments
///
/// * `expr` - The expression to differentiate
/// * `vars` - The variable names
/// * `direction` - The direction vector (must have same length as vars)
///
/// # Returns
///
/// The directional derivative as a symbolic expression
pub fn directional_derivative(expr: &Expr, vars: &[&str], direction: &[Expr]) -> Result<Expr> {
    if vars.len() != direction.len() {
        return Err(NumRs2Error::ValueError(
            "Direction vector must have same length as variable list".to_string(),
        ));
    }

    let grad = gradient(expr, vars)?;

    if grad.is_empty() {
        return Ok(Expr::Constant(0.0));
    }

    // Compute dot product: ∇f · v
    let mut result = Expr::Mul(Box::new(grad[0].clone()), Box::new(direction[0].clone()));

    for i in 1..grad.len() {
        result = Expr::Add(
            Box::new(result),
            Box::new(Expr::Mul(
                Box::new(grad[i].clone()),
                Box::new(direction[i].clone()),
            )),
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_differentiate_constant() {
        let c = Expr::constant(5.0);
        let dc = differentiate(&c, "x").expect("differentiation failed");

        assert!(matches!(dc, Expr::Constant(0.0)));
    }

    #[test]
    fn test_differentiate_variable() {
        let x = Expr::var("x");
        let dx = differentiate(&x, "x").expect("differentiation failed");

        assert!(matches!(dx, Expr::Constant(1.0)));

        let dy = differentiate(&x, "y").expect("differentiation failed");
        assert!(matches!(dy, Expr::Constant(0.0)));
    }

    #[test]
    fn test_differentiate_sum() {
        let x = Expr::var("x");
        let expr = x.clone() + x.clone();
        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(x + x) = 1 + 1 = 2
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_differentiate_product() {
        let x = Expr::var("x");
        let expr = x.clone() * x.clone(); // x²
        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(x²) = 2x, at x=3: 2*3 = 6
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_differentiate_power() {
        let x = Expr::var("x");
        let expr = x.clone().pow(3.0); // x³
        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(x³) = 3x², at x=2: 3*4 = 12
        assert_eq!(result, 12.0);
    }

    #[test]
    fn test_differentiate_sin() {
        let x = Expr::var("x");
        let expr = x.clone().sin();
        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(sin(x)) = cos(x), at x=0: cos(0) = 1
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_differentiate_exp() {
        let x = Expr::var("x");
        let expr = x.clone().exp();
        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(exp(x)) = exp(x), at x=0: exp(0) = 1
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_gradient() {
        let x = Expr::var("x");
        let y = Expr::var("y");
        let f = x.clone() * x.clone() + y.clone() * y.clone(); // x² + y²

        let grad = gradient(&f, &["x", "y"]).expect("gradient computation failed");
        assert_eq!(grad.len(), 2);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        vars.insert("y".to_string(), 4.0);

        let dx = grad[0].eval(&vars).expect("evaluation failed");
        let dy = grad[1].eval(&vars).expect("evaluation failed");

        // ∇f = [2x, 2y], at (3, 4): [6, 8]
        assert_eq!(dx, 6.0);
        assert_eq!(dy, 8.0);
    }

    #[test]
    fn test_chain_rule() {
        let x = Expr::var("x");
        let inner = x.clone() * 2.0;
        let expr = inner.sin(); // sin(2x)

        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(sin(2x)) = cos(2x) * 2, at x=0: cos(0) * 2 = 2
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_quotient_rule() {
        let x = Expr::var("x");
        let expr = x.clone() / (x.clone() + 1.0); // x / (x + 1)

        let derivative = differentiate(&expr, "x").expect("differentiation failed");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let result = derivative.eval(&vars).expect("evaluation failed");

        // d/dx(x/(x+1)) = ((x+1) - x) / (x+1)² = 1 / (x+1)²
        // at x=1: 1/4 = 0.25
        assert_eq!(result, 0.25);
    }
}
