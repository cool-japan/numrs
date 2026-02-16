//! Expression simplification and algebraic transformations
//!
//! This module implements algebraic simplification rules to reduce symbolic
//! expressions to simpler forms.

use crate::symbolic::expr::Expr;

/// Simplify a symbolic expression using algebraic rules
///
/// Applies various simplification rules including:
/// - Constant folding: 2 + 3 → 5
/// - Identity operations: x + 0 → x, x * 1 → x, x * 0 → 0
/// - Algebraic rules: x - x → 0, x / x → 1
/// - Negation: --x → x, -0 → 0
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let expr = x.clone() + 0.0; // x + 0
/// let simplified = simplify(&expr);
/// // Result: x
/// ```
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Constant(_) | Expr::Variable(_) => expr.clone(),

        // Addition simplification
        Expr::Add(f, g) => {
            let f_simp = simplify(f);
            let g_simp = simplify(g);

            // Constant folding
            if let (Expr::Constant(a), Expr::Constant(b)) = (&f_simp, &g_simp) {
                return Expr::Constant(a + b);
            }

            // x + 0 = x
            if matches!(g_simp, Expr::Constant(c) if c == 0.0) {
                return f_simp;
            }

            // 0 + x = x
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return g_simp;
            }

            Expr::Add(Box::new(f_simp), Box::new(g_simp))
        }

        // Subtraction simplification
        Expr::Sub(f, g) => {
            let f_simp = simplify(f);
            let g_simp = simplify(g);

            // Constant folding
            if let (Expr::Constant(a), Expr::Constant(b)) = (&f_simp, &g_simp) {
                return Expr::Constant(a - b);
            }

            // x - 0 = x
            if matches!(g_simp, Expr::Constant(c) if c == 0.0) {
                return f_simp;
            }

            // 0 - x = -x
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Neg(Box::new(g_simp));
            }

            // x - x = 0 (simple case)
            if f_simp == g_simp {
                return Expr::Constant(0.0);
            }

            Expr::Sub(Box::new(f_simp), Box::new(g_simp))
        }

        // Multiplication simplification
        Expr::Mul(f, g) => {
            let f_simp = simplify(f);
            let g_simp = simplify(g);

            // Constant folding
            if let (Expr::Constant(a), Expr::Constant(b)) = (&f_simp, &g_simp) {
                return Expr::Constant(a * b);
            }

            // x * 0 = 0
            if matches!(f_simp, Expr::Constant(c) if c == 0.0)
                || matches!(g_simp, Expr::Constant(c) if c == 0.0)
            {
                return Expr::Constant(0.0);
            }

            // x * 1 = x
            if matches!(g_simp, Expr::Constant(c) if c == 1.0) {
                return f_simp;
            }

            // 1 * x = x
            if matches!(f_simp, Expr::Constant(c) if c == 1.0) {
                return g_simp;
            }

            // x * -1 = -x
            if matches!(g_simp, Expr::Constant(c) if c == -1.0) {
                return Expr::Neg(Box::new(f_simp));
            }

            // -1 * x = -x
            if matches!(f_simp, Expr::Constant(c) if c == -1.0) {
                return Expr::Neg(Box::new(g_simp));
            }

            Expr::Mul(Box::new(f_simp), Box::new(g_simp))
        }

        // Division simplification
        Expr::Div(f, g) => {
            let f_simp = simplify(f);
            let g_simp = simplify(g);

            // Constant folding (careful with division by zero)
            if let (Expr::Constant(a), Expr::Constant(b)) = (&f_simp, &g_simp) {
                if *b != 0.0 {
                    return Expr::Constant(a / b);
                }
            }

            // 0 / x = 0 (if x != 0)
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(0.0);
            }

            // x / 1 = x
            if matches!(g_simp, Expr::Constant(c) if c == 1.0) {
                return f_simp;
            }

            // x / x = 1 (simple case)
            if f_simp == g_simp {
                return Expr::Constant(1.0);
            }

            // x / -1 = -x
            if matches!(g_simp, Expr::Constant(c) if c == -1.0) {
                return Expr::Neg(Box::new(f_simp));
            }

            Expr::Div(Box::new(f_simp), Box::new(g_simp))
        }

        // Power simplification
        Expr::Pow(f, g) => {
            let f_simp = simplify(f);
            let g_simp = simplify(g);

            // Constant folding
            if let (Expr::Constant(a), Expr::Constant(b)) = (&f_simp, &g_simp) {
                if *a >= 0.0 || b.fract() == 0.0 {
                    return Expr::Constant(a.powf(*b));
                }
            }

            // x^0 = 1
            if matches!(g_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(1.0);
            }

            // x^1 = x
            if matches!(g_simp, Expr::Constant(c) if c == 1.0) {
                return f_simp;
            }

            // 0^x = 0 (for x > 0)
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                if let Expr::Constant(exp) = g_simp {
                    if exp > 0.0 {
                        return Expr::Constant(0.0);
                    }
                }
            }

            // 1^x = 1
            if matches!(f_simp, Expr::Constant(c) if c == 1.0) {
                return Expr::Constant(1.0);
            }

            Expr::Pow(Box::new(f_simp), Box::new(g_simp))
        }

        // Negation simplification
        Expr::Neg(f) => {
            let f_simp = simplify(f);

            // -(-x) = x
            if let Expr::Neg(inner) = f_simp {
                return *inner;
            }

            // -0 = 0
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(0.0);
            }

            // Constant folding
            if let Expr::Constant(c) = f_simp {
                return Expr::Constant(-c);
            }

            Expr::Neg(Box::new(f_simp))
        }

        // Trigonometric function simplification
        Expr::Sin(f) => {
            let f_simp = simplify(f);

            // sin(0) = 0
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(0.0);
            }

            // Constant folding
            if let Expr::Constant(c) = f_simp {
                return Expr::Constant(c.sin());
            }

            Expr::Sin(Box::new(f_simp))
        }

        Expr::Cos(f) => {
            let f_simp = simplify(f);

            // cos(0) = 1
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(1.0);
            }

            // Constant folding
            if let Expr::Constant(c) = f_simp {
                return Expr::Constant(c.cos());
            }

            Expr::Cos(Box::new(f_simp))
        }

        Expr::Tan(f) => {
            let f_simp = simplify(f);

            // tan(0) = 0
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(0.0);
            }

            // Constant folding
            if let Expr::Constant(c) = f_simp {
                return Expr::Constant(c.tan());
            }

            Expr::Tan(Box::new(f_simp))
        }

        // Exponential and logarithm simplification
        Expr::Exp(f) => {
            let f_simp = simplify(f);

            // exp(0) = 1
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(1.0);
            }

            // Constant folding
            if let Expr::Constant(c) = f_simp {
                return Expr::Constant(c.exp());
            }

            // exp(ln(x)) = x
            if let Expr::Ln(inner) = &f_simp {
                return (**inner).clone();
            }

            Expr::Exp(Box::new(f_simp))
        }

        Expr::Ln(f) => {
            let f_simp = simplify(f);

            // ln(1) = 0
            if matches!(f_simp, Expr::Constant(c) if c == 1.0) {
                return Expr::Constant(0.0);
            }

            // Constant folding (careful with ln of negative numbers)
            if let Expr::Constant(c) = f_simp {
                if c > 0.0 {
                    return Expr::Constant(c.ln());
                }
            }

            // ln(exp(x)) = x
            if let Expr::Exp(inner) = &f_simp {
                return (**inner).clone();
            }

            Expr::Ln(Box::new(f_simp))
        }

        Expr::Sqrt(f) => {
            let f_simp = simplify(f);

            // sqrt(0) = 0
            if matches!(f_simp, Expr::Constant(c) if c == 0.0) {
                return Expr::Constant(0.0);
            }

            // sqrt(1) = 1
            if matches!(f_simp, Expr::Constant(c) if c == 1.0) {
                return Expr::Constant(1.0);
            }

            // Constant folding (careful with negative numbers)
            if let Expr::Constant(c) = f_simp {
                if c >= 0.0 {
                    return Expr::Constant(c.sqrt());
                }
            }

            // sqrt(x^2) = |x| (we'll just keep it as sqrt for simplicity)

            Expr::Sqrt(Box::new(f_simp))
        }
    }
}

/// Expand products and powers in an expression
///
/// Applies distributive law to expand expressions like (x+1)*(x+2).
///
/// Note: This is a basic implementation that handles simple cases.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let expr = (x.clone() + 1.0) * (x.clone() + 2.0);
/// let expanded = expand(&expr);
/// // Result: x² + 3x + 2 (in expanded form)
/// ```
pub fn expand(expr: &Expr) -> Expr {
    match expr {
        // Base cases
        Expr::Constant(_) | Expr::Variable(_) => expr.clone(),

        // Expand addition/subtraction recursively
        Expr::Add(f, g) => Expr::Add(Box::new(expand(f)), Box::new(expand(g))),

        Expr::Sub(f, g) => Expr::Sub(Box::new(expand(f)), Box::new(expand(g))),

        // Distributive law: (a + b) * c = a*c + b*c
        Expr::Mul(f, g) => {
            let f_exp = expand(f);
            let g_exp = expand(g);

            // (f1 + f2) * g = f1*g + f2*g
            if let Expr::Add(f1, f2) = f_exp {
                return expand(&Expr::Add(
                    Box::new(Expr::Mul(f1, Box::new(g_exp.clone()))),
                    Box::new(Expr::Mul(f2, Box::new(g_exp))),
                ));
            }

            // f * (g1 + g2) = f*g1 + f*g2
            if let Expr::Add(g1, g2) = g_exp {
                return expand(&Expr::Add(
                    Box::new(Expr::Mul(Box::new(f_exp.clone()), g1)),
                    Box::new(Expr::Mul(Box::new(f_exp), g2)),
                ));
            }

            // (f1 - f2) * g = f1*g - f2*g
            if let Expr::Sub(f1, f2) = f_exp {
                return expand(&Expr::Sub(
                    Box::new(Expr::Mul(f1, Box::new(g_exp.clone()))),
                    Box::new(Expr::Mul(f2, Box::new(g_exp))),
                ));
            }

            // f * (g1 - g2) = f*g1 - f*g2
            if let Expr::Sub(g1, g2) = g_exp {
                return expand(&Expr::Sub(
                    Box::new(Expr::Mul(Box::new(f_exp.clone()), g1)),
                    Box::new(Expr::Mul(Box::new(f_exp), g2)),
                ));
            }

            Expr::Mul(Box::new(f_exp), Box::new(g_exp))
        }

        // Expand powers: (a + b)^2 = (a + b) * (a + b), then expand
        Expr::Pow(f, g) => {
            let f_exp = expand(f);

            // Special case: (expr)^n where n is a small positive integer
            if let Expr::Constant(n) = **g {
                if n > 0.0 && n <= 5.0 && n.fract() == 0.0 {
                    let n_int = n as i32;
                    let mut result = f_exp.clone();
                    for _ in 1..n_int {
                        result = expand(&Expr::Mul(Box::new(result), Box::new(f_exp.clone())));
                    }
                    return result;
                }
            }

            Expr::Pow(Box::new(f_exp), Box::new(expand(g)))
        }

        // Other operations
        Expr::Div(f, g) => Expr::Div(Box::new(expand(f)), Box::new(expand(g))),
        Expr::Neg(f) => Expr::Neg(Box::new(expand(f))),
        Expr::Sin(f) => Expr::Sin(Box::new(expand(f))),
        Expr::Cos(f) => Expr::Cos(Box::new(expand(f))),
        Expr::Tan(f) => Expr::Tan(Box::new(expand(f))),
        Expr::Exp(f) => Expr::Exp(Box::new(expand(f))),
        Expr::Ln(f) => Expr::Ln(Box::new(expand(f))),
        Expr::Sqrt(f) => Expr::Sqrt(Box::new(expand(f))),
    }
}

/// Factor simple expressions (basic implementation)
///
/// This is a placeholder for more advanced factoring algorithms.
/// Currently handles only simple cases.
pub fn factor(expr: &Expr) -> Expr {
    // For now, just return the simplified expression
    // A full factoring implementation would be quite complex
    simplify(expr)
}

/// Check if two expressions are structurally equal
///
/// This is more strict than semantic equality - it checks if the
/// expression trees have the same structure.
pub fn structurally_equal(a: &Expr, b: &Expr) -> bool {
    a == b
}

/// Collect like terms in an expression (basic implementation)
///
/// This would combine terms like 2x + 3x into 5x.
/// Currently just simplifies the expression.
pub fn collect_terms(expr: &Expr) -> Expr {
    simplify(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_addition() {
        let x = Expr::var("x");

        // x + 0 = x
        let expr = x.clone() + 0.0;
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 0 + x = x
        let expr = 0.0 + x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 2 + 3 = 5
        let expr = Expr::constant(2.0) + Expr::constant(3.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(5.0));
    }

    #[test]
    fn test_simplify_multiplication() {
        let x = Expr::var("x");

        // x * 0 = 0
        let expr = x.clone() * 0.0;
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(0.0));

        // x * 1 = x
        let expr = x.clone() * 1.0;
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 2 * 3 = 6
        let expr = Expr::constant(2.0) * Expr::constant(3.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(6.0));
    }

    #[test]
    fn test_simplify_power() {
        let x = Expr::var("x");

        // x^0 = 1
        let expr = x.clone().pow(0.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(1.0));

        // x^1 = x
        let expr = x.clone().pow(1.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 2^3 = 8
        let expr = Expr::constant(2.0).pow(3.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(8.0));
    }

    #[test]
    fn test_simplify_negation() {
        let x = Expr::var("x");

        // -(-x) = x
        let expr = -(-x.clone());
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // -0 = 0
        let expr = -Expr::constant(0.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(0.0));
    }

    #[test]
    fn test_simplify_exp_ln() {
        let x = Expr::var("x");

        // exp(ln(x)) = x
        let expr = x.clone().ln().exp();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // ln(exp(x)) = x
        let expr = x.clone().exp().ln();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_expand_distributive() {
        let x = Expr::var("x");

        // (x + 1) * 2 = 2x + 2
        let expr = (x.clone() + 1.0) * 2.0;
        let expanded = expand(&expr);
        let simplified = simplify(&expanded);

        // Check by evaluation
        use std::collections::HashMap;
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);

        let original_val = expr.eval(&vars).expect("eval failed");
        let expanded_val = simplified.eval(&vars).expect("eval failed");

        assert_eq!(original_val, expanded_val);
    }

    #[test]
    fn test_simplify_division() {
        let x = Expr::var("x");

        // x / 1 = x
        let expr = x.clone() / 1.0;
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 0 / x = 0
        let expr = 0.0 / x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(0.0));

        // 6 / 2 = 3
        let expr = Expr::constant(6.0) / Expr::constant(2.0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::constant(3.0));
    }
}
