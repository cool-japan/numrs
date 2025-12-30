# NumRS Polynomial Module

The NumRS polynomial module provides functionality for working with polynomials, including creation, evaluation, arithmetic operations, root finding, fitting, and interpolation. This module is designed to offer functionality similar to NumPy's polynomial capabilities with Rust's safety and performance.

## Key Features

- Polynomial representation and operations
- Polynomial evaluation and arithmetic
- Polynomial fitting (least squares)
- Root finding for polynomials
- Various interpolation methods (Lagrange, cubic spline, etc.)
- Chebyshev, Legendre, and other orthogonal polynomials
- Integration and differentiation of polynomials

## Basic Usage

### Creating and Evaluating Polynomials

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{Polynomial, polyval};

// Create a polynomial p(x) = 1 + 2x + 3x²
let p = Polynomial::new(vec![1.0, 2.0, 3.0]);

// Evaluate the polynomial at x = 2.0
let value = p.evaluate(2.0); // 1 + 2*2 + 3*2² = 1 + 4 + 12 = 17
println!("p(2.0) = {}", value);

// Evaluate for multiple x values using an array
let x = Array::linspace(0.0, 2.0, 5)?;  // [0.0, 0.5, 1.0, 1.5, 2.0]
let values = polyval(&p.coefficients(), &x)?;
println!("Polynomial evaluated at multiple points: {}", values);
```

### Polynomial Arithmetic

```rust
use numrs2::prelude::*;
use numrs2::polynomial::Polynomial;

// Create two polynomials: p(x) = 1 + 2x + 3x² and q(x) = 4 + 5x
let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
let q = Polynomial::new(vec![4.0, 5.0]);

// Addition: (1 + 2x + 3x²) + (4 + 5x) = 5 + 7x + 3x²
let sum = &p + &q;
println!("p + q = {}", sum);

// Subtraction: (1 + 2x + 3x²) - (4 + 5x) = -3 - 3x + 3x²
let diff = &p - &q;
println!("p - q = {}", diff);

// Multiplication: (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
let product = &p * &q;
println!("p * q = {}", product);

// Scalar operations
let scaled = &p * 2.0;  // 2 + 4x + 6x²
println!("2 * p = {}", scaled);
```

### Polynomial Fitting

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{polyfit, Polynomial};

// Create data points
let x = Array::linspace(0.0, 4.0, 5)?;  // [0.0, 1.0, 2.0, 3.0, 4.0]
let y = Array::from_vec(vec![1.0, 6.0, 17.0, 34.0, 57.0]);  // Values of 1 + 2x + 3x²

// Fit a 2nd degree polynomial
let coeffs = polyfit(&x, &y, 2)?;
let p_fit = Polynomial::new(coeffs.to_vec());
println!("Fitted polynomial: {}", p_fit);

// Evaluate the fitted polynomial at new points
let x_new = Array::linspace(0.5, 3.5, 4)?;
let y_pred = p_fit.evaluate_array(&x_new)?;
println!("Predictions at new points: {}", y_pred);
```

### Root Finding

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{Polynomial, roots};

// Create a polynomial p(x) = x² - 5x + 6 (with roots 2 and 3)
let p = Polynomial::new(vec![6.0, -5.0, 1.0]);

// Find the roots
let roots_vec = roots(&p.coefficients())?;
println!("Roots of p(x): {}", roots_vec);  // Should be close to [2.0, 3.0]

// We can also find roots of higher-degree polynomials
// For example, p(x) = (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
let higher_degree = Polynomial::new(vec![-6.0, 11.0, -6.0, 1.0]);
let complex_roots = roots(&higher_degree.coefficients())?;
println!("Roots of higher-degree polynomial: {}", complex_roots);
```

## Interpolation Methods

### Lagrange Interpolation

```rust
use numrs2::prelude::*;
use numrs2::polynomial::interpolation::{lagrange_interpolation, LagrangeInterpolation};

// Create data points
let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
let y = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0]);

// Create a Lagrange interpolation
let interpolator = LagrangeInterpolation::new(&x, &y)?;

// Evaluate at a point
let value = interpolator.evaluate(1.5);
println!("Interpolated value at x=1.5: {}", value);

// Evaluate at multiple points
let x_new = Array::linspace(0.0, 3.0, 7)?;
let y_new = lagrange_interpolation(&x, &y, &x_new)?;
println!("Interpolated values: {}", y_new);
```

### Cubic Spline Interpolation

```rust
use numrs2::prelude::*;
use numrs2::polynomial::interpolation::{CubicSpline, spline_interpolation};

// Create data points
let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
let y = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0]);

// Create a cubic spline
let spline = CubicSpline::new(&x, &y)?;

// Evaluate at a point
let value = spline.evaluate(1.5);
println!("Spline value at x=1.5: {}", value);

// Evaluate at multiple points
let x_new = Array::linspace(0.0, 3.0, 7)?;
let y_new = spline_interpolation(&x, &y, &x_new)?;
println!("Spline values: {}", y_new);
```

### Polynomial Interpolation

```rust
use numrs2::prelude::*;
use numrs2::polynomial::interpolation::{PolynomialInterpolation, polynomial_interpolation};

// Create data points
let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
let y = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0]);

// Create a polynomial interpolation
let interpolator = PolynomialInterpolation::new(&x, &y)?;

// Evaluate at a point
let value = interpolator.evaluate(1.5);
println!("Interpolated value at x=1.5: {}", value);

// Evaluate at multiple points
let x_new = Array::linspace(0.0, 3.0, 7)?;
let y_new = polynomial_interpolation(&x, &y, &x_new)?;
println!("Interpolated values: {}", y_new);
```

## Orthogonal Polynomials

### Chebyshev Polynomials

```rust
use numrs2::prelude::*;
use numrs2::polynomial::orthogonal::{chebyshev_polynomial, evaluate_chebyshev};

// Generate a Chebyshev polynomial of first kind, degree 3
let cheby_t3 = chebyshev_polynomial(3, false)?;
println!("Chebyshev T3(x): {}", cheby_t3);

// Generate a Chebyshev polynomial of second kind, degree 2
let cheby_u2 = chebyshev_polynomial(2, true)?;
println!("Chebyshev U2(x): {}", cheby_u2);

// Evaluate Chebyshev polynomials at points
let x = Array::linspace(-1.0, 1.0, 5)?;
let values = evaluate_chebyshev(&x, 3, false)?;
println!("T3(x) evaluated at points: {}", values);
```

### Legendre Polynomials

```rust
use numrs2::prelude::*;
use numrs2::polynomial::orthogonal::{legendre_polynomial, evaluate_legendre};

// Generate a Legendre polynomial of degree 2: P₂(x) = (3x² - 1)/2
let legendre_p2 = legendre_polynomial(2)?;
println!("Legendre P2(x): {}", legendre_p2);

// Evaluate Legendre polynomials at points
let x = Array::linspace(-1.0, 1.0, 5)?;
let values = evaluate_legendre(&x, 2)?;
println!("P2(x) evaluated at points: {}", values);
```

## Integration and Differentiation

### Polynomial Differentiation

```rust
use numrs2::prelude::*;
use numrs2::polynomial::Polynomial;

// Create a polynomial p(x) = 1 + 2x + 3x²
let p = Polynomial::new(vec![1.0, 2.0, 3.0]);

// Differentiate: p'(x) = 2 + 6x
let derivative = p.derivative();
println!("p'(x) = {}", derivative);

// Second derivative: p''(x) = 6
let second_derivative = derivative.derivative();
println!("p''(x) = {}", second_derivative);
```

### Polynomial Integration

```rust
use numrs2::prelude::*;
use numrs2::polynomial::Polynomial;

// Create a polynomial p(x) = 1 + 2x + 3x²
let p = Polynomial::new(vec![1.0, 2.0, 3.0]);

// Integrate: ∫p(x)dx = C + x + x² + x³
let integral = p.integral();
println!("∫p(x)dx = {}", integral);

// Definite integral: ∫₀¹ p(x)dx
let definite_integral = p.integrate(0.0, 1.0);
println!("∫₀¹ p(x)dx = {}", definite_integral);
```

## Polynomial Classes and Conversion

### Polynomial and Power Series

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{Polynomial, PowerSeries};

// Create a polynomial with standard basis
let poly = Polynomial::new(vec![1.0, 2.0, 3.0]);  // 1 + 2x + 3x²

// Convert to power series (same representation for standard polynomials)
let series = PowerSeries::from_polynomial(&poly);
println!("Power series: {}", series);

// Evaluate both at a point
let x = 2.0;
println!("p({}) = {}", x, poly.evaluate(x));
println!("s({}) = {}", x, series.evaluate(x));
```

### Specialized Polynomial Types

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{
    Polynomial, ChebyshevSeries, LegendreSeries, polynomial_to_chebyshev, polynomial_to_legendre
};

// Create a polynomial with standard basis
let poly = Polynomial::new(vec![1.0, 0.0, -1.0]);  // 1 - x²

// Convert to Chebyshev series
let cheby = polynomial_to_chebyshev(&poly.coefficients())?;
let cheby_series = ChebyshevSeries::new(cheby);
println!("Chebyshev series: {}", cheby_series);

// Convert to Legendre series
let legendre = polynomial_to_legendre(&poly.coefficients())?;
let legendre_series = LegendreSeries::new(legendre);
println!("Legendre series: {}", legendre_series);
```

## Applications

### Curve Fitting

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{polyfit, polyval};

// Generate noisy data from a polynomial
let x = Array::linspace(0.0, 10.0, 20)?;
let y_true = x.power(2.0).multiply_scalar(2.0).add_scalar(-3.0);  // 2x² - 3

// Add some noise
let mut rng = numrs2::random::default_rng();
let noise = rng.normal(0.0, 1.0, &[20])?;
let y_noisy = y_true.add(&noise);

// Fit polynomials of different degrees
let linear_coeffs = polyfit(&x, &y_noisy, 1)?;
let quadratic_coeffs = polyfit(&x, &y_noisy, 2)?;
let cubic_coeffs = polyfit(&x, &y_noisy, 3)?;

// Evaluate on a finer grid
let x_fine = Array::linspace(0.0, 10.0, 100)?;
let linear_pred = polyval(&linear_coeffs, &x_fine)?;
let quadratic_pred = polyval(&quadratic_coeffs, &x_fine)?;
let cubic_pred = polyval(&cubic_coeffs, &x_fine)?;

println!("True coefficients: [2.0, 0.0, -3.0]");
println!("Linear fit coefficients: {}", linear_coeffs);
println!("Quadratic fit coefficients: {}", quadratic_coeffs);
println!("Cubic fit coefficients: {}", cubic_coeffs);
```

### Root Finding Applications

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{Polynomial, roots};

// Find the intersection points of two polynomials
// p(x) = x² - 2  and  q(x) = x - 2
let p = Polynomial::new(vec![-2.0, 0.0, 1.0]);  // x² - 2
let q = Polynomial::new(vec![-2.0, 1.0]);       // x - 2

// Subtract to find where they are equal: p(x) - q(x) = 0
let diff = &p - &q;
println!("p(x) - q(x) = {}", diff);  // x² - x

// Find the roots
let intersection_points = roots(&diff.coefficients())?;
println!("Intersection points: {}", intersection_points);  // Should be [0.0, 1.0]

// Verify
let p_at_points = p.evaluate_array(&intersection_points)?;
let q_at_points = q.evaluate_array(&intersection_points)?;
println!("p evaluated at intersection points: {}", p_at_points);
println!("q evaluated at intersection points: {}", q_at_points);
```

## Examples

For detailed examples of polynomial operations, see:
- `polynomial_example.rs`: Basic polynomial operations and fitting
- `interpolation_example.rs`: Various interpolation methods

## Implementation Details

- NumRS polynomials use coefficient-based representation
- Operations are optimized for numerical stability
- Special basis functions (Chebyshev, Legendre) are supported for precision
- Evaluation uses Horner's method for efficiency
- Integration with other NumRS modules for extended functionality