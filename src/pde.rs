//! Partial Differential Equation (PDE) Solvers
//!
//! This module provides numerical methods for solving partial differential equations:
//!
//! - **Finite Difference Methods**: 1D and 2D heat equation, wave equation
//! - **Method of Lines**: Convert PDEs to systems of ODEs
//! - **Boundary Conditions**: Dirichlet, Neumann, periodic
//!
//! # Supported PDEs
//!
//! - Heat equation (diffusion): ∂u/∂t = α∇²u
//! - Wave equation: ∂²u/∂t² = c²∇²u
//! - Poisson equation: ∇²u = f
//! - Laplace equation: ∇²u = 0
//!
//! # Example
//!
//! ```ignore
//! use numrs2::pde::{solve_heat_1d, BoundaryCondition};
//!
//! // Solve 1D heat equation with Dirichlet boundary conditions
//! let initial = vec![0.0; 50];
//! let result = solve_heat_1d(
//!     &initial,
//!     0.1,  // alpha (thermal diffusivity)
//!     0.01, // dx (spatial step)
//!     0.001, // dt (time step)
//!     100,  // number of time steps
//!     BoundaryCondition::Dirichlet(0.0),
//!     BoundaryCondition::Dirichlet(1.0),
//! ).expect("valid PDE parameters");
//! ```

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Boundary condition types for PDE solvers
#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition<T> {
    /// Dirichlet boundary condition: u = value at boundary
    Dirichlet(T),
    /// Neumann boundary condition: ∂u/∂n = value at boundary
    Neumann(T),
    /// Periodic boundary condition: u wraps around
    Periodic,
}

/// Result of PDE solution
#[derive(Debug, Clone)]
pub struct PdeResult<T> {
    /// Solution values at each time step
    pub u: Vec<Vec<T>>,
    /// Time values corresponding to each solution
    pub t: Vec<T>,
    /// Spatial grid points
    pub x: Vec<T>,
    /// Number of spatial points
    pub nx: usize,
    /// Number of time steps
    pub nt: usize,
    /// Success flag
    pub success: bool,
    /// Message
    pub message: String,
}

/// Result of 2D PDE solution
#[derive(Debug, Clone)]
pub struct Pde2dResult<T> {
    /// Solution values at each time step (flattened 2D grid)
    pub u: Vec<Vec<T>>,
    /// Time values
    pub t: Vec<T>,
    /// X grid points
    pub x: Vec<T>,
    /// Y grid points
    pub y: Vec<T>,
    /// Number of X points
    pub nx: usize,
    /// Number of Y points
    pub ny: usize,
    /// Number of time steps
    pub nt: usize,
    /// Success flag
    pub success: bool,
}

/// Solve 1D heat equation using explicit finite difference (FTCS scheme)
///
/// Solves: ∂u/∂t = α ∂²u/∂x²
///
/// Uses Forward-Time Central-Space (FTCS) discretization:
/// `u[n+1][i] = u[n][i] + r*(u[n][i+1] - 2*u[n][i] + u[n][i-1])`
/// where `r = α*dt/dx²`
///
/// # Arguments
/// * `initial` - Initial condition u(x, 0)
/// * `alpha` - Thermal diffusivity coefficient
/// * `dx` - Spatial step size
/// * `dt` - Time step size
/// * `nt` - Number of time steps
/// * `bc_left` - Left boundary condition
/// * `bc_right` - Right boundary condition
///
/// # Stability
/// Requires r = α*dt/dx² ≤ 0.5 for stability
///
/// # Returns
/// * `PdeResult` with solution at each time step
pub fn solve_heat_1d<T>(
    initial: &[T],
    alpha: T,
    dx: T,
    dt: T,
    nt: usize,
    bc_left: BoundaryCondition<T>,
    bc_right: BoundaryCondition<T>,
) -> Result<PdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
{
    let nx = initial.len();
    if nx < 3 {
        return Err(NumRs2Error::ValueError(
            "Need at least 3 spatial points".to_string(),
        ));
    }

    // Compute r = α*dt/dx²
    let r = alpha * dt / (dx * dx);

    // Check stability (CFL condition)
    let half = T::from(0.5).expect("0.5 is representable as Float");
    if r > half {
        return Err(NumRs2Error::ValueError(format!(
            "Unstable: r = {:?} > 0.5. Reduce dt or increase dx.",
            r
        )));
    }

    // Initialize solution
    let mut u = vec![initial.to_vec()];
    let mut current = initial.to_vec();
    let mut next = vec![T::zero(); nx];

    // Time values
    let mut t = vec![T::zero()];

    // Generate x grid
    let x: Vec<T> = (0..nx)
        .map(|i| T::from(i).expect("i is representable as Float") * dx)
        .collect();

    // Time stepping
    for n in 0..nt {
        // Apply boundary conditions
        apply_bc_1d(&mut current, bc_left, bc_right, dx);

        // Update interior points using FTCS
        for i in 1..nx - 1 {
            let d2u = current[i + 1]
                - T::from(2.0).expect("2.0 is representable as Float") * current[i]
                + current[i - 1];
            next[i] = current[i] + r * d2u;
        }

        // Apply boundary conditions to next step
        apply_bc_to_next(&mut next, bc_left, bc_right, &current, dx);

        // Store and swap
        current = next.clone();
        u.push(current.clone());
        t.push(T::from(n + 1).expect("n+1 is representable as Float") * dt);
    }

    Ok(PdeResult {
        u,
        t,
        x,
        nx,
        nt: nt + 1,
        success: true,
        message: "Heat equation solved successfully".to_string(),
    })
}

/// Apply boundary conditions to a 1D array
fn apply_bc_1d<T: Float + Debug>(
    u: &mut [T],
    bc_left: BoundaryCondition<T>,
    bc_right: BoundaryCondition<T>,
    dx: T,
) {
    let n = u.len();

    match bc_left {
        BoundaryCondition::Dirichlet(val) => u[0] = val,
        BoundaryCondition::Neumann(val) => u[0] = u[1] - dx * val,
        BoundaryCondition::Periodic => u[0] = u[n - 2],
    }

    match bc_right {
        BoundaryCondition::Dirichlet(val) => u[n - 1] = val,
        BoundaryCondition::Neumann(val) => u[n - 1] = u[n - 2] + dx * val,
        BoundaryCondition::Periodic => u[n - 1] = u[1],
    }
}

/// Apply boundary conditions to next time step
fn apply_bc_to_next<T: Float + Debug>(
    next: &mut [T],
    bc_left: BoundaryCondition<T>,
    bc_right: BoundaryCondition<T>,
    current: &[T],
    dx: T,
) {
    let n = next.len();

    match bc_left {
        BoundaryCondition::Dirichlet(val) => next[0] = val,
        BoundaryCondition::Neumann(val) => next[0] = next[1] - dx * val,
        BoundaryCondition::Periodic => next[0] = next[n - 2],
    }

    match bc_right {
        BoundaryCondition::Dirichlet(val) => next[n - 1] = val,
        BoundaryCondition::Neumann(val) => next[n - 1] = next[n - 2] + dx * val,
        BoundaryCondition::Periodic => next[n - 1] = next[1],
    }
}

/// Solve 1D heat equation using implicit Crank-Nicolson scheme
///
/// More stable than explicit FTCS, unconditionally stable.
/// Uses tridiagonal matrix algorithm (Thomas algorithm) to solve.
///
/// # Arguments
/// * `initial` - Initial condition
/// * `alpha` - Thermal diffusivity
/// * `dx` - Spatial step
/// * `dt` - Time step
/// * `nt` - Number of time steps
/// * `bc_left` - Left boundary condition
/// * `bc_right` - Right boundary condition
pub fn solve_heat_1d_crank_nicolson<T>(
    initial: &[T],
    alpha: T,
    dx: T,
    dt: T,
    nt: usize,
    bc_left: BoundaryCondition<T>,
    bc_right: BoundaryCondition<T>,
) -> Result<PdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
{
    let nx = initial.len();
    if nx < 3 {
        return Err(NumRs2Error::ValueError(
            "Need at least 3 spatial points".to_string(),
        ));
    }

    let r = alpha * dt / (dx * dx);
    let half = T::from(0.5).expect("0.5 is representable as Float");
    let two = T::from(2.0).expect("2.0 is representable as Float");
    let one = T::one();

    // Coefficients for tridiagonal system
    // A*u^{n+1} = B*u^n
    // A = [1+r, -r/2, 0, ...]
    //     [-r/2, 1+r, -r/2, ...]
    //     ...
    // B = [1-r, r/2, 0, ...]
    //     [r/2, 1-r, r/2, ...]
    //     ...

    let a_diag = one + r; // Main diagonal of A
    let a_off = -half * r; // Off-diagonal of A
    let b_diag = one - r; // Main diagonal of B
    let b_off = half * r; // Off-diagonal of B

    let mut u = vec![initial.to_vec()];
    let mut current = initial.to_vec();
    let mut t = vec![T::zero()];
    let x: Vec<T> = (0..nx)
        .map(|i| T::from(i).expect("i is representable as Float") * dx)
        .collect();

    for n in 0..nt {
        // Apply boundary conditions
        apply_bc_1d(&mut current, bc_left, bc_right, dx);

        // Build right-hand side: B*u^n
        let mut rhs = vec![T::zero(); nx - 2];
        for i in 1..nx - 1 {
            rhs[i - 1] = b_off * current[i - 1] + b_diag * current[i] + b_off * current[i + 1];
        }

        // Adjust for boundary conditions
        if let BoundaryCondition::Dirichlet(val) = bc_left {
            rhs[0] = rhs[0] - a_off * val;
        }
        if let BoundaryCondition::Dirichlet(val) = bc_right {
            rhs[nx - 3] = rhs[nx - 3] - a_off * val;
        }

        // Solve tridiagonal system using Thomas algorithm
        let solution = thomas_algorithm(
            &vec![a_off; nx - 3],
            &vec![a_diag; nx - 2],
            &vec![a_off; nx - 3],
            &rhs,
        )?;

        // Update current solution
        current[1..(nx - 1)].copy_from_slice(&solution[..((nx - 1) - 1)]);
        apply_bc_1d(&mut current, bc_left, bc_right, dx);

        u.push(current.clone());
        t.push(T::from(n + 1).expect("n+1 is representable as Float") * dt);
    }

    Ok(PdeResult {
        u,
        t,
        x,
        nx,
        nt: nt + 1,
        success: true,
        message: "Crank-Nicolson heat equation solved successfully".to_string(),
    })
}

/// Thomas algorithm for solving tridiagonal systems
fn thomas_algorithm<T>(a: &[T], b: &[T], c: &[T], d: &[T]) -> Result<Vec<T>>
where
    T: Float + Debug,
{
    let n = b.len();
    if a.len() != n - 1 || c.len() != n - 1 || d.len() != n {
        return Err(NumRs2Error::ValueError(
            "Invalid sizes for tridiagonal system".to_string(),
        ));
    }

    // Forward elimination
    let mut c_prime = vec![T::zero(); n - 1];
    let mut d_prime = vec![T::zero(); n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n - 1 {
        let denom = b[i] - a[i - 1] * c_prime[i - 1];
        if denom.abs() < T::epsilon() {
            return Err(NumRs2Error::ValueError(
                "Division by zero in Thomas algorithm".to_string(),
            ));
        }
        c_prime[i] = c[i] / denom;
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom;
    }

    // Last element
    let denom = b[n - 1] - a[n - 2] * c_prime[n - 2];
    d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / denom;

    // Back substitution
    let mut x = vec![T::zero(); n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solve 1D wave equation using explicit finite difference
///
/// Solves: ∂²u/∂t² = c² ∂²u/∂x²
///
/// # Arguments
/// * `initial_u` - Initial displacement u(x, 0)
/// * `initial_v` - Initial velocity ∂u/∂t(x, 0)
/// * `c` - Wave speed
/// * `dx` - Spatial step
/// * `dt` - Time step
/// * `nt` - Number of time steps
/// * `bc_left` - Left boundary condition
/// * `bc_right` - Right boundary condition
///
/// # Stability
/// Requires c*dt/dx ≤ 1 (Courant condition)
pub fn solve_wave_1d<T>(
    initial_u: &[T],
    initial_v: &[T],
    c: T,
    dx: T,
    dt: T,
    nt: usize,
    bc_left: BoundaryCondition<T>,
    bc_right: BoundaryCondition<T>,
) -> Result<PdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
{
    let nx = initial_u.len();
    if nx < 3 {
        return Err(NumRs2Error::ValueError(
            "Need at least 3 spatial points".to_string(),
        ));
    }
    if initial_v.len() != nx {
        return Err(NumRs2Error::DimensionMismatch(
            "Initial u and v must have same length".to_string(),
        ));
    }

    // Courant number
    let courant = c * dt / dx;
    if courant > T::one() {
        return Err(NumRs2Error::ValueError(format!(
            "Unstable: Courant number {:?} > 1. Reduce dt or increase dx.",
            courant
        )));
    }

    let courant_sq = courant * courant;
    let two = T::from(2.0).expect("2.0 is representable as Float");

    // Initialize
    let mut u_curr = initial_u.to_vec();
    let mut u_prev; // Will be set after first step

    // First time step uses initial velocity
    let mut u_next = vec![T::zero(); nx];
    for i in 1..nx - 1 {
        let d2u = u_curr[i + 1] - two * u_curr[i] + u_curr[i - 1];
        u_next[i] = u_curr[i] + dt * initial_v[i] + courant_sq * d2u / two;
    }
    apply_bc_1d(&mut u_next, bc_left, bc_right, dx);

    let mut result_u = vec![u_curr.clone(), u_next.clone()];
    let mut t = vec![T::zero(), dt];
    let x: Vec<T> = (0..nx)
        .map(|i| T::from(i).expect("i is representable as Float") * dx)
        .collect();

    u_prev = u_curr;
    u_curr = u_next.clone();

    // Continue time stepping
    for n in 1..nt {
        for i in 1..nx - 1 {
            let d2u = u_curr[i + 1] - two * u_curr[i] + u_curr[i - 1];
            u_next[i] = two * u_curr[i] - u_prev[i] + courant_sq * d2u;
        }
        apply_bc_1d(&mut u_next, bc_left, bc_right, dx);

        u_prev = u_curr.clone();
        u_curr = u_next.clone();
        result_u.push(u_curr.clone());
        t.push(T::from(n + 1).expect("n+1 is representable as Float") * dt);
    }

    Ok(PdeResult {
        u: result_u,
        t,
        x,
        nx,
        nt: nt + 1,
        success: true,
        message: "Wave equation solved successfully".to_string(),
    })
}

/// Solve 2D heat equation using explicit finite difference
///
/// Solves: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
///
/// # Arguments
/// * `initial` - Initial condition (flattened row-major, nx*ny elements)
/// * `nx` - Number of x grid points
/// * `ny` - Number of y grid points
/// * `alpha` - Thermal diffusivity
/// * `dx` - X spatial step
/// * `dy` - Y spatial step
/// * `dt` - Time step
/// * `nt` - Number of time steps
/// * `bc` - Boundary condition (applied to all boundaries)
///
/// # Stability
/// Requires α*dt*(1/dx² + 1/dy²) ≤ 0.5
pub fn solve_heat_2d<T>(
    initial: &[T],
    nx: usize,
    ny: usize,
    alpha: T,
    dx: T,
    dy: T,
    dt: T,
    nt: usize,
    bc: BoundaryCondition<T>,
) -> Result<Pde2dResult<T>>
where
    T: Float + Debug + std::iter::Sum,
{
    if initial.len() != nx * ny {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Initial condition size {} doesn't match nx*ny={}",
            initial.len(),
            nx * ny
        )));
    }
    if nx < 3 || ny < 3 {
        return Err(NumRs2Error::ValueError(
            "Need at least 3 points in each dimension".to_string(),
        ));
    }

    let rx = alpha * dt / (dx * dx);
    let ry = alpha * dt / (dy * dy);
    let r_total = rx + ry;

    let half = T::from(0.5).expect("0.5 is representable as Float");
    if r_total > half {
        return Err(NumRs2Error::ValueError(format!(
            "Unstable: r = {:?} > 0.5. Reduce dt or increase dx/dy.",
            r_total
        )));
    }

    let two = T::from(2.0).expect("2.0 is representable as Float");

    let mut u_curr = initial.to_vec();
    let mut u_next = vec![T::zero(); nx * ny];
    let mut result_u = vec![u_curr.clone()];
    let mut t = vec![T::zero()];

    let x: Vec<T> = (0..nx)
        .map(|i| T::from(i).expect("i is representable as Float") * dx)
        .collect();
    let y: Vec<T> = (0..ny)
        .map(|j| T::from(j).expect("j is representable as Float") * dy)
        .collect();

    for n in 0..nt {
        // Apply boundary conditions
        apply_bc_2d(&mut u_curr, nx, ny, bc);

        // Update interior points
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let idx = j * nx + i;
                let d2ux = u_curr[idx + 1] - two * u_curr[idx] + u_curr[idx - 1];
                let d2uy = u_curr[idx + nx] - two * u_curr[idx] + u_curr[idx - nx];
                u_next[idx] = u_curr[idx] + rx * d2ux + ry * d2uy;
            }
        }

        // Apply boundary conditions to next step
        apply_bc_2d(&mut u_next, nx, ny, bc);

        u_curr = u_next.clone();
        result_u.push(u_curr.clone());
        t.push(T::from(n + 1).expect("n+1 is representable as Float") * dt);
    }

    Ok(Pde2dResult {
        u: result_u,
        t,
        x,
        y,
        nx,
        ny,
        nt: nt + 1,
        success: true,
    })
}

/// Apply boundary conditions to 2D array
fn apply_bc_2d<T: Float>(u: &mut [T], nx: usize, ny: usize, bc: BoundaryCondition<T>) {
    match bc {
        BoundaryCondition::Dirichlet(val) => {
            // Top and bottom rows
            for i in 0..nx {
                u[i] = val; // Bottom (j=0)
                u[(ny - 1) * nx + i] = val; // Top (j=ny-1)
            }
            // Left and right columns
            for j in 0..ny {
                u[j * nx] = val; // Left (i=0)
                u[j * nx + nx - 1] = val; // Right (i=nx-1)
            }
        }
        BoundaryCondition::Neumann(_) => {
            // Zero-gradient: copy from adjacent interior
            for i in 1..nx - 1 {
                u[i] = u[nx + i]; // Bottom
                u[(ny - 1) * nx + i] = u[(ny - 2) * nx + i]; // Top
            }
            for j in 1..ny - 1 {
                u[j * nx] = u[j * nx + 1]; // Left
                u[j * nx + nx - 1] = u[j * nx + nx - 2]; // Right
            }
            // Corners
            u[0] = u[nx + 1];
            u[nx - 1] = u[nx + nx - 2];
            u[(ny - 1) * nx] = u[(ny - 2) * nx + 1];
            u[ny * nx - 1] = u[(ny - 1) * nx - 2];
        }
        BoundaryCondition::Periodic => {
            // Wrap around
            for i in 0..nx {
                u[i] = u[(ny - 2) * nx + i]; // Bottom = Top-1
                u[(ny - 1) * nx + i] = u[nx + i]; // Top = Bottom+1
            }
            for j in 0..ny {
                u[j * nx] = u[j * nx + nx - 2]; // Left = Right-1
                u[j * nx + nx - 1] = u[j * nx + 1]; // Right = Left+1
            }
        }
    }
}

/// Solve Poisson equation using Jacobi iteration
///
/// Solves: ∇²u = f
///
/// # Arguments
/// * `f` - Source term (flattened row-major)
/// * `nx` - Number of x grid points
/// * `ny` - Number of y grid points
/// * `dx` - X spatial step
/// * `dy` - Y spatial step
/// * `bc` - Boundary condition
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// Solution u and convergence information
pub fn solve_poisson_2d<T>(
    f: &[T],
    nx: usize,
    ny: usize,
    dx: T,
    dy: T,
    bc: BoundaryCondition<T>,
    max_iter: usize,
    tol: T,
) -> Result<(Vec<T>, usize, T)>
where
    T: Float + Debug + std::iter::Sum,
{
    if f.len() != nx * ny {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Source size {} doesn't match nx*ny={}",
            f.len(),
            nx * ny
        )));
    }

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let two = T::from(2.0).expect("2.0 is representable as Float");
    let denom = two * (dx2 + dy2);

    let mut u = vec![T::zero(); nx * ny];
    let mut u_new = vec![T::zero(); nx * ny];

    for iter in 0..max_iter {
        apply_bc_2d(&mut u, nx, ny, bc);

        let mut max_diff = T::zero();

        // Jacobi iteration
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let idx = j * nx + i;
                let u_xx = u[idx + 1] + u[idx - 1];
                let u_yy = u[idx + nx] + u[idx - nx];
                u_new[idx] = (dy2 * u_xx + dx2 * u_yy - dx2 * dy2 * f[idx]) / denom;

                let diff = (u_new[idx] - u[idx]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        // Check convergence
        if max_diff < tol {
            return Ok((u_new, iter + 1, max_diff));
        }

        std::mem::swap(&mut u, &mut u_new);
    }

    Ok((u, max_iter, T::infinity()))
}

/// Solve Poisson equation using Gauss-Seidel iteration (faster convergence)
pub fn solve_poisson_gauss_seidel<T>(
    f: &[T],
    nx: usize,
    ny: usize,
    dx: T,
    dy: T,
    bc: BoundaryCondition<T>,
    max_iter: usize,
    tol: T,
) -> Result<(Vec<T>, usize, T)>
where
    T: Float + Debug + std::iter::Sum,
{
    if f.len() != nx * ny {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Source size {} doesn't match nx*ny={}",
            f.len(),
            nx * ny
        )));
    }

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let two = T::from(2.0).expect("2.0 is representable as Float");
    let denom = two * (dx2 + dy2);

    let mut u = vec![T::zero(); nx * ny];

    for iter in 0..max_iter {
        apply_bc_2d(&mut u, nx, ny, bc);

        let mut max_diff = T::zero();

        // Gauss-Seidel iteration (in-place update)
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let idx = j * nx + i;
                let u_xx = u[idx + 1] + u[idx - 1];
                let u_yy = u[idx + nx] + u[idx - nx];
                let u_old = u[idx];
                u[idx] = (dy2 * u_xx + dx2 * u_yy - dx2 * dy2 * f[idx]) / denom;

                let diff = (u[idx] - u_old).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        // Check convergence
        if max_diff < tol {
            return Ok((u, iter + 1, max_diff));
        }
    }

    Ok((u, max_iter, T::infinity()))
}

/// Solve Poisson equation using SOR (Successive Over-Relaxation)
pub fn solve_poisson_sor<T>(
    f: &[T],
    nx: usize,
    ny: usize,
    dx: T,
    dy: T,
    bc: BoundaryCondition<T>,
    omega: T,
    max_iter: usize,
    tol: T,
) -> Result<(Vec<T>, usize, T)>
where
    T: Float + Debug + std::iter::Sum,
{
    if f.len() != nx * ny {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Source size {} doesn't match nx*ny={}",
            f.len(),
            nx * ny
        )));
    }

    if omega <= T::zero() || omega >= T::from(2.0).expect("2.0 is representable as Float") {
        return Err(NumRs2Error::ValueError(
            "Omega must be in (0, 2) for SOR convergence".to_string(),
        ));
    }

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let two = T::from(2.0).expect("2.0 is representable as Float");
    let denom = two * (dx2 + dy2);
    let one_minus_omega = T::one() - omega;

    let mut u = vec![T::zero(); nx * ny];

    for iter in 0..max_iter {
        apply_bc_2d(&mut u, nx, ny, bc);

        let mut max_diff = T::zero();

        // SOR iteration
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let idx = j * nx + i;
                let u_xx = u[idx + 1] + u[idx - 1];
                let u_yy = u[idx + nx] + u[idx - nx];
                let u_old = u[idx];
                let u_gs = (dy2 * u_xx + dx2 * u_yy - dx2 * dy2 * f[idx]) / denom;
                u[idx] = one_minus_omega * u_old + omega * u_gs;

                let diff = (u[idx] - u_old).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        if max_diff < tol {
            return Ok((u, iter + 1, max_diff));
        }
    }

    Ok((u, max_iter, T::infinity()))
}

/// Compute optimal SOR relaxation parameter for Laplace equation
pub fn optimal_omega<T>(nx: usize, ny: usize) -> T
where
    T: Float,
{
    let pi = T::from(std::f64::consts::PI).expect("PI is representable as Float");
    let n = T::from(nx.min(ny)).expect("nx.min(ny) is representable as Float");
    let rho = (pi / (n + T::one())).cos();
    T::from(2.0).expect("2.0 is representable as Float")
        / (T::one() + (T::one() - rho * rho).sqrt())
}

/// Method of Lines: Convert 1D PDE to system of ODEs
///
/// Discretizes spatial derivatives, leaving time continuous.
/// Returns a function that can be used with ODE solvers.
///
/// For heat equation: du_i/dt = α/dx² * (u_{i+1} - 2u_i + u_{i-1})
pub fn method_of_lines_heat<T>(
    nx: usize,
    alpha: T,
    dx: T,
    bc_left: BoundaryCondition<T>,
    bc_right: BoundaryCondition<T>,
) -> impl Fn(T, &[T]) -> Vec<T>
where
    T: Float + Debug,
{
    let coef = alpha / (dx * dx);
    let two = T::from(2.0).expect("2.0 is representable as Float");

    move |_t: T, u: &[T]| -> Vec<T> {
        let mut dudt = vec![T::zero(); nx];

        // Interior points
        for i in 1..nx - 1 {
            dudt[i] = coef * (u[i + 1] - two * u[i] + u[i - 1]);
        }

        // Boundary conditions
        match bc_left {
            BoundaryCondition::Dirichlet(_) => dudt[0] = T::zero(),
            BoundaryCondition::Neumann(val) => {
                // Ghost point approach
                let u_ghost =
                    u[1] - T::from(2.0).expect("2.0 is representable as Float") * dx * val;
                dudt[0] = coef * (u[1] - two * u[0] + u_ghost);
            }
            BoundaryCondition::Periodic => {
                dudt[0] = coef * (u[1] - two * u[0] + u[nx - 2]);
            }
        }

        match bc_right {
            BoundaryCondition::Dirichlet(_) => dudt[nx - 1] = T::zero(),
            BoundaryCondition::Neumann(val) => {
                let u_ghost =
                    u[nx - 2] + T::from(2.0).expect("2.0 is representable as Float") * dx * val;
                dudt[nx - 1] = coef * (u_ghost - two * u[nx - 1] + u[nx - 2]);
            }
            BoundaryCondition::Periodic => {
                dudt[nx - 1] = coef * (u[1] - two * u[nx - 1] + u[nx - 2]);
            }
        }

        dudt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_1d_basic() {
        // Initial condition: step function
        let mut initial = vec![0.0f64; 50];
        for i in 20..30 {
            initial[i] = 1.0;
        }

        let result = solve_heat_1d(
            &initial,
            0.1,
            0.02,
            0.001,
            100,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        )
        .expect("solve_heat_1d should succeed with valid parameters");

        assert!(result.success);
        assert_eq!(result.u.len(), 101);
        assert_eq!(result.nx, 50);
    }

    #[test]
    fn test_heat_1d_stability() {
        // Should fail with unstable parameters
        let initial = vec![0.0f64; 50];
        let result = solve_heat_1d(
            &initial,
            1.0,  // Large alpha
            0.01, // Small dx
            0.1,  // Large dt
            10,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_heat_1d_crank_nicolson() {
        let initial: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();

        let result = solve_heat_1d_crank_nicolson(
            &initial,
            0.1,
            0.02,
            0.01, // Larger dt is OK for Crank-Nicolson
            50,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        )
        .expect("Crank-Nicolson should succeed with valid parameters");

        assert!(result.success);
    }

    #[test]
    fn test_wave_1d_basic() {
        // Initial Gaussian pulse
        let nx = 100;
        let initial_u: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 - 50.0;
                (-x * x / 100.0).exp()
            })
            .collect();
        let initial_v = vec![0.0f64; nx];

        let result = solve_wave_1d(
            &initial_u,
            &initial_v,
            1.0,  // c
            0.1,  // dx
            0.05, // dt (CFL = 0.5)
            100,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        )
        .expect("solve_wave_1d should succeed with valid parameters");

        assert!(result.success);
        assert_eq!(result.u.len(), 101);
    }

    #[test]
    fn test_wave_1d_courant() {
        // Should fail with CFL > 1
        let initial_u = vec![0.0f64; 50];
        let initial_v = vec![0.0f64; 50];

        let result = solve_wave_1d(
            &initial_u,
            &initial_v,
            1.0,
            0.1,
            0.2, // CFL = 2 > 1
            10,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_heat_2d_basic() {
        let nx = 20;
        let ny = 20;
        let mut initial = vec![0.0f64; nx * ny];
        // Hot spot in center
        for j in 8..12 {
            for i in 8..12 {
                initial[j * nx + i] = 1.0;
            }
        }

        let result = solve_heat_2d(
            &initial,
            nx,
            ny,
            0.1,
            0.1,
            0.1,
            0.01,
            50,
            BoundaryCondition::Dirichlet(0.0),
        )
        .expect("solve_heat_2d should succeed with valid parameters");

        assert!(result.success);
        assert_eq!(result.u.len(), 51);
    }

    #[test]
    fn test_poisson_jacobi() {
        let nx = 20;
        let ny = 20;
        let f = vec![0.0f64; nx * ny]; // Laplace equation

        let (solution, _iters, _error) = solve_poisson_2d(
            &f,
            nx,
            ny,
            0.1,
            0.1,
            BoundaryCondition::Dirichlet(0.0),
            1000,
            1e-6,
        )
        .expect("solve_poisson_2d should succeed with valid parameters");

        assert_eq!(solution.len(), nx * ny);
        // For Laplace with zero boundary, solution should be zero
        assert!(solution.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_poisson_gauss_seidel() {
        let nx = 20;
        let ny = 20;
        let f = vec![1.0f64; nx * ny]; // Non-zero source

        let (solution, _iters, _) = solve_poisson_gauss_seidel(
            &f,
            nx,
            ny,
            0.1,
            0.1,
            BoundaryCondition::Dirichlet(0.0),
            500,
            1e-6,
        )
        .expect("solve_poisson_gauss_seidel should succeed with valid parameters");

        assert_eq!(solution.len(), nx * ny);
        // Solution should be non-zero in interior
        let center = solution[ny / 2 * nx + nx / 2];
        assert!(center.abs() > 0.0);
    }

    #[test]
    fn test_poisson_sor() {
        let nx = 30;
        let ny = 30;
        let f = vec![0.0f64; nx * ny];

        let omega = optimal_omega::<f64>(nx, ny);
        assert!(omega > 1.0 && omega < 2.0);

        let (_, iters_sor, _) = solve_poisson_sor(
            &f,
            nx,
            ny,
            0.1,
            0.1,
            BoundaryCondition::Dirichlet(0.0),
            omega,
            500,
            1e-6,
        )
        .expect("solve_poisson_sor should succeed with valid parameters");

        // SOR should converge faster than Jacobi
        // (Just check it completes)
        assert!(iters_sor > 0);
    }

    #[test]
    fn test_thomas_algorithm() {
        // Test solving [2, -1, 0] [x1]   [1]
        //              [-1, 2, -1] [x2] = [0]
        //              [0, -1, 2]  [x3]   [1]
        let a = vec![-1.0f64, -1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![-1.0f64, -1.0];
        let d = vec![1.0, 0.0, 1.0];

        let x = thomas_algorithm(&a, &b, &c, &d)
            .expect("thomas_algorithm should succeed with valid input");

        assert_eq!(x.len(), 3);
        // Verify solution
        let tol = 1e-10;
        assert!((2.0 * x[0] - x[1] - 1.0).abs() < tol);
        assert!((-x[0] + 2.0 * x[1] - x[2]).abs() < tol);
        assert!((-x[1] + 2.0 * x[2] - 1.0).abs() < tol);
    }

    #[test]
    fn test_method_of_lines() {
        let nx = 10;
        let alpha = 0.1;
        let dx = 0.1;

        let f = method_of_lines_heat(
            nx,
            alpha,
            dx,
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Dirichlet(0.0),
        );

        // Test with uniform initial condition
        let u = vec![1.0f64; nx];
        let dudt = f(0.0, &u);

        assert_eq!(dudt.len(), nx);
        // Boundary points should have zero derivative (Dirichlet)
        assert_eq!(dudt[0], 0.0);
        assert_eq!(dudt[nx - 1], 0.0);
    }

    #[test]
    fn test_periodic_boundary() {
        let initial: Vec<f64> = (0..50)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin())
            .collect();

        let result = solve_heat_1d(
            &initial,
            0.1,
            0.02,
            0.001,
            100,
            BoundaryCondition::Periodic,
            BoundaryCondition::Periodic,
        )
        .expect("solve_heat_1d with periodic BC should succeed");

        assert!(result.success);
    }

    #[test]
    fn test_neumann_boundary() {
        let initial = vec![1.0f64; 50];

        let result = solve_heat_1d(
            &initial,
            0.1,
            0.02,
            0.001,
            100,
            BoundaryCondition::Neumann(0.0),
            BoundaryCondition::Neumann(0.0),
        )
        .expect("solve_heat_1d with Neumann BC should succeed");

        assert!(result.success);
        // With zero Neumann (insulated) and uniform IC, solution should stay uniform
        let last = result.u.last().expect("solution should have elements");
        let mean: f64 = last.iter().sum::<f64>() / last.len() as f64;
        assert!((mean - 1.0).abs() < 0.1);
    }
}
