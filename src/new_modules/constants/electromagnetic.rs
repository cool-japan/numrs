//! Electromagnetic Physical Constants (CODATA 2022)
//!
//! This module provides electromagnetic physical constants as recommended
//! by the Committee on Data of the International Science Council (CODATA) 2022.
//!
//! These constants characterize the electromagnetic interaction in vacuum,
//! including permittivity, permeability, impedance, and quantum electromagnetic
//! effects (flux quantum, Josephson and von Klitzing constants).
//!
//! # Note on Post-2019 SI Values
//!
//! After the 2019 SI redefinition, the vacuum permittivity and permeability
//! are no longer exact but are experimentally determined quantities (with
//! very small uncertainties). This is because the elementary charge and
//! Planck constant are now exact, which propagates uncertainty to epsilon_0
//! and mu_0.
//!
//! # References
//!
//! - CODATA 2022: <https://physics.nist.gov/cuu/Constants/>
//! - E. Tiesinga, P.J. Mohr, D.B. Newell, and B.N. Taylor,
//!   "CODATA recommended values of the fundamental physical constants: 2022",
//!   Rev. Mod. Phys. (2024).

use super::PhysicalConstant;

/// Vacuum electric permittivity (epsilon_0)
///
/// Value: 8.854_187_818_8 x 10^-12 F/m
/// Uncertainty: 1.4 x 10^-21 F/m
///
/// The vacuum permittivity (also called the electric constant or permittivity
/// of free space) relates the electric displacement field to the electric field
/// in vacuum. After the 2019 SI redefinition, it is no longer exact but is
/// determined by epsilon_0 = 1 / (mu_0 * c^2) = e^2 / (2 * alpha * h * c).
pub const VACUUM_PERMITTIVITY: PhysicalConstant = PhysicalConstant {
    value: 8.854_187_818_8e-12,
    uncertainty: 1.4e-21,
    unit: "F/m",
    symbol: "\u{03B5}_0", // Unicode ε_0
    name: "Vacuum electric permittivity",
};

/// Vacuum magnetic permeability (mu_0)
///
/// Value: 1.256_637_062_12 x 10^-6 N/A^2
/// Uncertainty: 1.9 x 10^-16 N/A^2
///
/// The vacuum permeability (also called the magnetic constant or permeability
/// of free space) relates the magnetic induction field to the magnetic field
/// intensity in vacuum. After the 2019 SI redefinition, it is no longer
/// exactly 4*pi*10^-7 but is determined by mu_0 = 2*alpha*h / (e^2 * c).
pub const VACUUM_PERMEABILITY: PhysicalConstant = PhysicalConstant {
    value: 1.256_637_062_12e-6,
    uncertainty: 1.9e-16,
    unit: "N/A^2",
    symbol: "\u{03BC}_0", // Unicode μ_0
    name: "Vacuum magnetic permeability",
};

/// Impedance of free space (Z_0)
///
/// Value: 376.730_313_412 ohm
/// Uncertainty: 5.9 x 10^-8 ohm
///
/// The impedance of free space is the ratio of the electric field magnitude
/// to the magnetic field magnitude in an electromagnetic plane wave in vacuum.
/// It is defined as Z_0 = mu_0 * c. In the pre-2019 SI, Z_0 was exactly
/// 4*pi/c * 10^-7 * c = mu_0 * c. Now it has a small uncertainty.
pub const IMPEDANCE_OF_FREE_SPACE: PhysicalConstant = PhysicalConstant {
    value: 376.730_313_412,
    uncertainty: 5.9e-8,
    unit: "\u{03A9}", // Unicode Ω (ohm)
    symbol: "Z_0",
    name: "Impedance of free space",
};

/// Coulomb constant (k_e)
///
/// Value: 8.987_551_792_3 x 10^9 N m^2 C^-2
/// Uncertainty: 1.4 x 10^0 N m^2 C^-2
///
/// The Coulomb constant (also called the electric force constant) appears
/// in Coulomb's law: F = k_e * q1 * q2 / r^2. It is defined as
/// k_e = 1 / (4 * pi * epsilon_0).
pub const COULOMB_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 8.987_551_792_3e9,
    uncertainty: 1.4,
    unit: "N m^2 C^-2",
    symbol: "k_e",
    name: "Coulomb constant",
};

/// Magnetic flux quantum (Phi_0)
///
/// Value: 2.067_833_848... x 10^-15 Wb (exact, derived)
///
/// The magnetic flux quantum is the quantum of magnetic flux passing through
/// a superconductor. It is defined as Phi_0 = h / (2*e). Since both h and e
/// are exact in the 2019 SI, the magnetic flux quantum is also exact.
pub const MAGNETIC_FLUX_QUANTUM: PhysicalConstant = PhysicalConstant {
    // Phi_0 = h / (2*e) = 6.626_070_15e-34 / (2 * 1.602_176_634e-19)
    // = 2.067_833_848_461_929_3e-15
    value: 2.067_833_848_461_929_3e-15,
    uncertainty: 0.0,
    unit: "Wb",
    symbol: "\u{03A6}_0", // Unicode Φ_0
    name: "Magnetic flux quantum",
};

/// Conductance quantum (G_0)
///
/// Value: 7.748_091_729... x 10^-5 S (exact, derived)
///
/// The conductance quantum is the quantized unit of electrical conductance.
/// It is defined as G_0 = 2*e^2 / h. It appears in the quantized conductance
/// of quantum point contacts and quantum wires. Since both e and h are exact,
/// G_0 is also exact.
pub const CONDUCTANCE_QUANTUM: PhysicalConstant = PhysicalConstant {
    // G_0 = 2*e^2/h = 2 * (1.602_176_634e-19)^2 / 6.626_070_15e-34
    // = 7.748_091_729_863_649e-5
    value: 7.748_091_729_863_649e-5,
    uncertainty: 0.0,
    unit: "S",
    symbol: "G_0",
    name: "Conductance quantum",
};

/// Josephson constant (K_J)
///
/// Value: 483_597.848_4... x 10^9 Hz/V (exact, derived)
///
/// The Josephson constant relates the frequency of the AC Josephson effect
/// to the voltage across a Josephson junction: f = K_J * V. It is defined
/// as K_J = 2*e/h. Since both e and h are exact in the 2019 SI, K_J is
/// also exact.
pub const JOSEPHSON_CONSTANT: PhysicalConstant = PhysicalConstant {
    // K_J = 2*e/h = 2 * 1.602_176_634e-19 / 6.626_070_15e-34
    // = 4.835_978_484_169_836_3e14
    value: 4.835_978_484_169_836_3e14,
    uncertainty: 0.0,
    unit: "Hz/V",
    symbol: "K_J",
    name: "Josephson constant",
};

/// Von Klitzing constant (R_K)
///
/// Value: 25_812.807_45... ohm (exact, derived)
///
/// The von Klitzing constant is the quantized Hall resistance observed in
/// the integer quantum Hall effect. It is defined as R_K = h / e^2.
/// Since both h and e are exact in the 2019 SI, R_K is also exact.
/// Note: R_K = 1 / G_0 * 2 (since G_0 = 2*e^2/h).
pub const VON_KLITZING_CONSTANT: PhysicalConstant = PhysicalConstant {
    // R_K = h/e^2 = 6.626_070_15e-34 / (1.602_176_634e-19)^2
    // = 25_812.807_459_304_513
    value: 25_812.807_459_304_513,
    uncertainty: 0.0,
    unit: "\u{03A9}", // Unicode Ω (ohm)
    symbol: "R_K",
    name: "Von Klitzing constant",
};

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;
    use crate::new_modules::constants::fundamental::{
        ELEMENTARY_CHARGE, PLANCK_CONSTANT, SPEED_OF_LIGHT,
    };

    const REL_TOL: f64 = 1e-5;

    #[test]
    fn test_vacuum_permittivity_value() {
        let expected = 8.854_187_818_8e-12;
        assert!((VACUUM_PERMITTIVITY.value - expected).abs() < 1e-21);
    }

    #[test]
    fn test_vacuum_permeability_value() {
        let expected = 1.256_637_062_12e-6;
        assert!((VACUUM_PERMEABILITY.value - expected).abs() < 1e-16);
    }

    #[test]
    fn test_impedance_of_free_space_value() {
        let expected = 376.730_313_412;
        assert!((IMPEDANCE_OF_FREE_SPACE.value - expected).abs() < 1e-7);
    }

    #[test]
    fn test_coulomb_constant_value() {
        let expected = 8.987_551_792_3e9;
        assert!((COULOMB_CONSTANT.value - expected).abs() < 1.0);
    }

    #[test]
    fn test_magnetic_flux_quantum_value() {
        let expected = 2.067_833_848_461_929_3e-15;
        assert!((MAGNETIC_FLUX_QUANTUM.value - expected).abs() < 1e-28);
        assert!(MAGNETIC_FLUX_QUANTUM.is_exact());
    }

    #[test]
    fn test_conductance_quantum_value() {
        let expected = 7.748_091_729_863_649e-5;
        assert!((CONDUCTANCE_QUANTUM.value - expected).abs() < 1e-18);
        assert!(CONDUCTANCE_QUANTUM.is_exact());
    }

    #[test]
    fn test_josephson_constant_value() {
        let expected = 4.835_978_484_169_836_3e14;
        assert!((JOSEPHSON_CONSTANT.value - expected).abs() < 1e1);
        assert!(JOSEPHSON_CONSTANT.is_exact());
    }

    #[test]
    fn test_von_klitzing_constant_value() {
        let expected = 25_812.807_459_304_513;
        assert!((VON_KLITZING_CONSTANT.value - expected).abs() < 1e-6);
        assert!(VON_KLITZING_CONSTANT.is_exact());
    }

    #[test]
    fn test_epsilon0_mu0_c2_equals_one() {
        // The fundamental relation: epsilon_0 * mu_0 * c^2 = 1
        let c = SPEED_OF_LIGHT.value;
        let product = VACUUM_PERMITTIVITY.value * VACUUM_PERMEABILITY.value * c * c;
        let relative_error = (product - 1.0).abs();
        assert!(
            relative_error < REL_TOL,
            "epsilon_0 * mu_0 * c^2 = {} (expected 1.0, error: {})",
            product,
            relative_error
        );
    }

    #[test]
    fn test_impedance_derived_from_mu0_c() {
        // Z_0 = mu_0 * c
        let expected = VACUUM_PERMEABILITY.value * SPEED_OF_LIGHT.value;
        let relative_error = ((IMPEDANCE_OF_FREE_SPACE.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Z_0 derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_coulomb_constant_derived() {
        // k_e = 1 / (4 * pi * epsilon_0)
        let expected = 1.0 / (4.0 * std::f64::consts::PI * VACUUM_PERMITTIVITY.value);
        let relative_error = ((COULOMB_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Coulomb constant derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_magnetic_flux_quantum_derived() {
        // Phi_0 = h / (2*e)
        let expected = PLANCK_CONSTANT.value / (2.0 * ELEMENTARY_CHARGE.value);
        let relative_error = ((MAGNETIC_FLUX_QUANTUM.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-12,
            "Phi_0 derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_conductance_quantum_derived() {
        // G_0 = 2*e^2/h
        let e = ELEMENTARY_CHARGE.value;
        let expected = 2.0 * e * e / PLANCK_CONSTANT.value;
        let relative_error = ((CONDUCTANCE_QUANTUM.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-12,
            "G_0 derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_josephson_constant_derived() {
        // K_J = 2*e/h
        let expected = 2.0 * ELEMENTARY_CHARGE.value / PLANCK_CONSTANT.value;
        let relative_error = ((JOSEPHSON_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-12,
            "K_J derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_von_klitzing_constant_derived() {
        // R_K = h / e^2
        let e = ELEMENTARY_CHARGE.value;
        let expected = PLANCK_CONSTANT.value / (e * e);
        let relative_error = ((VON_KLITZING_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-12,
            "R_K derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_von_klitzing_conductance_quantum_reciprocal() {
        // R_K = 2 / G_0 (since R_K = h/e^2 and G_0 = 2*e^2/h)
        let expected = 2.0 / CONDUCTANCE_QUANTUM.value;
        let relative_error = ((VON_KLITZING_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-12,
            "R_K vs 2/G_0 error: {}",
            relative_error
        );
    }

    #[test]
    fn test_exact_em_constants() {
        // After 2019 SI, Phi_0, G_0, K_J, R_K are exact
        assert!(MAGNETIC_FLUX_QUANTUM.is_exact());
        assert!(CONDUCTANCE_QUANTUM.is_exact());
        assert!(JOSEPHSON_CONSTANT.is_exact());
        assert!(VON_KLITZING_CONSTANT.is_exact());
    }

    #[test]
    fn test_nonexact_em_constants() {
        // epsilon_0, mu_0, Z_0, k_e are NOT exact in post-2019 SI
        assert!(!VACUUM_PERMITTIVITY.is_exact());
        assert!(!VACUUM_PERMEABILITY.is_exact());
        assert!(!IMPEDANCE_OF_FREE_SPACE.is_exact());
        assert!(!COULOMB_CONSTANT.is_exact());
    }

    #[test]
    fn test_all_em_values_positive() {
        assert!(VACUUM_PERMITTIVITY.value > 0.0);
        assert!(VACUUM_PERMEABILITY.value > 0.0);
        assert!(IMPEDANCE_OF_FREE_SPACE.value > 0.0);
        assert!(COULOMB_CONSTANT.value > 0.0);
        assert!(MAGNETIC_FLUX_QUANTUM.value > 0.0);
        assert!(CONDUCTANCE_QUANTUM.value > 0.0);
        assert!(JOSEPHSON_CONSTANT.value > 0.0);
        assert!(VON_KLITZING_CONSTANT.value > 0.0);
    }
}
