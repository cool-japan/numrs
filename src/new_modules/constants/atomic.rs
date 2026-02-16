//! Atomic and Nuclear Physical Constants (CODATA 2022)
//!
//! This module provides atomic and nuclear physical constants as recommended
//! by the Committee on Data of the International Science Council (CODATA) 2022.
//!
//! These constants characterize atomic structure, nuclear physics, and
//! quantum electrodynamics. They include the Bohr radius, Rydberg constant,
//! magnetic moments, and related quantities.
//!
//! # References
//!
//! - CODATA 2022: <https://physics.nist.gov/cuu/Constants/>
//! - E. Tiesinga, P.J. Mohr, D.B. Newell, and B.N. Taylor,
//!   "CODATA recommended values of the fundamental physical constants: 2022",
//!   Rev. Mod. Phys. (2024).

use super::PhysicalConstant;

/// Bohr radius (a_0)
///
/// Value: 5.291_772_105_44 x 10^-11 m
/// Uncertainty: 8.2 x 10^-21 m
///
/// The Bohr radius is the most probable distance between the nucleus and
/// the electron in a hydrogen atom in its ground state. It is defined as
/// a_0 = hbar / (m_e * c * alpha) = 4*pi*epsilon_0*hbar^2 / (m_e * e^2).
pub const BOHR_RADIUS: PhysicalConstant = PhysicalConstant {
    value: 5.291_772_105_44e-11,
    uncertainty: 8.2e-21,
    unit: "m",
    symbol: "a_0",
    name: "Bohr radius",
};

/// Rydberg constant (R_infinity)
///
/// Value: 10_973_731.568_157 m^-1
/// Uncertainty: 1.2 x 10^-5 m^-1
///
/// The Rydberg constant represents the limiting value of the highest
/// wavenumber of any photon that can be emitted from the hydrogen atom.
/// It is defined as R_inf = alpha^2 * m_e * c / (2 * h).
pub const RYDBERG_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 10_973_731.568_157,
    uncertainty: 1.2e-5,
    unit: "m^-1",
    symbol: "R_inf",
    name: "Rydberg constant",
};

/// Hartree energy (E_h)
///
/// Value: 4.359_744_722_206_0 x 10^-18 J
/// Uncertainty: 4.8 x 10^-30 J
///
/// The Hartree energy is the natural unit of energy in atomic physics.
/// It is defined as E_h = 2 * R_inf * h * c = alpha^2 * m_e * c^2.
/// One Hartree is approximately 27.2 eV.
pub const HARTREE_ENERGY: PhysicalConstant = PhysicalConstant {
    value: 4.359_744_722_206_0e-18,
    uncertainty: 4.8e-30,
    unit: "J",
    symbol: "E_h",
    name: "Hartree energy",
};

/// Classical electron radius (r_e)
///
/// Value: 2.817_940_320_5 x 10^-15 m
/// Uncertainty: 1.3 x 10^-24 m
///
/// The classical electron radius, also known as the Lorentz radius or
/// Thomson scattering length, is defined as r_e = e^2 / (4*pi*epsilon_0*m_e*c^2).
/// Despite its name, it does not represent the actual size of the electron,
/// which is a point particle in quantum electrodynamics.
pub const CLASSICAL_ELECTRON_RADIUS: PhysicalConstant = PhysicalConstant {
    value: 2.817_940_320_5e-15,
    uncertainty: 1.3e-24,
    unit: "m",
    symbol: "r_e",
    name: "Classical electron radius",
};

/// Compton wavelength (lambda_C)
///
/// Value: 2.426_310_235_38 x 10^-12 m
/// Uncertainty: 7.6 x 10^-22 m
///
/// The Compton wavelength of the electron is defined as lambda_C = h / (m_e * c).
/// It represents the wavelength of a photon whose energy is equal to the
/// rest mass energy of the electron. It appears in the Compton scattering
/// formula and sets the scale for quantum effects.
pub const COMPTON_WAVELENGTH: PhysicalConstant = PhysicalConstant {
    value: 2.426_310_235_38e-12,
    uncertainty: 7.6e-22,
    unit: "m",
    symbol: "\u{03BB}_C", // Unicode λ_C
    name: "Compton wavelength",
};

/// Bohr magneton (mu_B)
///
/// Value: 9.274_010_065_7 x 10^-24 J T^-1
/// Uncertainty: 2.9 x 10^-33 J T^-1
///
/// The Bohr magneton is the natural unit of magnetic moment for the electron.
/// It is defined as mu_B = e * hbar / (2 * m_e). The electron's magnetic
/// moment is approximately one Bohr magneton (with small QED corrections).
pub const BOHR_MAGNETON: PhysicalConstant = PhysicalConstant {
    value: 9.274_010_065_7e-24,
    uncertainty: 2.9e-33,
    unit: "J T^-1",
    symbol: "\u{03BC}_B", // Unicode μ_B
    name: "Bohr magneton",
};

/// Nuclear magneton (mu_N)
///
/// Value: 5.050_783_739_3 x 10^-27 J T^-1
/// Uncertainty: 1.6 x 10^-36 J T^-1
///
/// The nuclear magneton is the natural unit of magnetic moment for nucleons.
/// It is defined as mu_N = e * hbar / (2 * m_p). It is approximately
/// 1/1836 of the Bohr magneton due to the proton-electron mass ratio.
pub const NUCLEAR_MAGNETON: PhysicalConstant = PhysicalConstant {
    value: 5.050_783_739_3e-27,
    uncertainty: 1.6e-36,
    unit: "J T^-1",
    symbol: "\u{03BC}_N", // Unicode μ_N
    name: "Nuclear magneton",
};

/// Proton magnetic moment (mu_p)
///
/// Value: 1.410_606_797_36 x 10^-26 J T^-1
/// Uncertainty: 6.0 x 10^-36 J T^-1
///
/// The magnetic moment of the proton. Unlike the electron, whose magnetic
/// moment is close to one Bohr magneton, the proton's magnetic moment is
/// approximately 2.793 nuclear magnetons, reflecting its composite (quark)
/// structure.
pub const PROTON_MAGNETIC_MOMENT: PhysicalConstant = PhysicalConstant {
    value: 1.410_606_797_36e-26,
    uncertainty: 6.0e-36,
    unit: "J T^-1",
    symbol: "\u{03BC}_p", // Unicode μ_p
    name: "Proton magnetic moment",
};

/// Electron g-factor (g_e)
///
/// Value: -2.002_319_304_362_56
/// Uncertainty: 3.5 x 10^-13
///
/// The electron g-factor relates the electron's magnetic moment to its
/// spin angular momentum. In the Dirac theory, g_e = -2 exactly, but
/// quantum electrodynamic corrections (the anomalous magnetic moment)
/// give a small deviation. The measurement of this quantity is one of the
/// most precise tests of QED.
pub const ELECTRON_G_FACTOR: PhysicalConstant = PhysicalConstant {
    value: -2.002_319_304_362_56,
    uncertainty: 3.5e-13,
    unit: "",
    symbol: "g_e",
    name: "Electron g-factor",
};

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;
    use crate::new_modules::constants::fundamental::{
        ELECTRON_MASS, ELEMENTARY_CHARGE, FINE_STRUCTURE_CONSTANT, PLANCK_CONSTANT,
        REDUCED_PLANCK_CONSTANT, SPEED_OF_LIGHT,
    };

    const REL_TOL: f64 = 1e-5;

    #[test]
    fn test_bohr_radius_value() {
        let expected = 5.291_772_105_44e-11;
        assert!((BOHR_RADIUS.value - expected).abs() < 1e-20);
    }

    #[test]
    fn test_rydberg_constant_value() {
        let expected = 10_973_731.568_157;
        assert!((RYDBERG_CONSTANT.value - expected).abs() < 1e-5);
    }

    #[test]
    fn test_hartree_energy_value() {
        let expected = 4.359_744_722_206_0e-18;
        assert!((HARTREE_ENERGY.value - expected).abs() < 1e-29);
    }

    #[test]
    fn test_classical_electron_radius_value() {
        let expected = 2.817_940_320_5e-15;
        assert!((CLASSICAL_ELECTRON_RADIUS.value - expected).abs() < 1e-24);
    }

    #[test]
    fn test_compton_wavelength_value() {
        let expected = 2.426_310_235_38e-12;
        assert!((COMPTON_WAVELENGTH.value - expected).abs() < 1e-21);
    }

    #[test]
    fn test_bohr_magneton_value() {
        let expected = 9.274_010_065_7e-24;
        assert!((BOHR_MAGNETON.value - expected).abs() < 1e-33);
    }

    #[test]
    fn test_nuclear_magneton_value() {
        let expected = 5.050_783_739_3e-27;
        assert!((NUCLEAR_MAGNETON.value - expected).abs() < 1e-36);
    }

    #[test]
    fn test_proton_magnetic_moment_value() {
        let expected = 1.410_606_797_36e-26;
        assert!((PROTON_MAGNETIC_MOMENT.value - expected).abs() < 1e-35);
    }

    #[test]
    fn test_electron_g_factor_value() {
        let expected = -2.002_319_304_362_56;
        assert!((ELECTRON_G_FACTOR.value - expected).abs() < 1e-12);
    }

    #[test]
    fn test_electron_g_factor_close_to_minus_two() {
        // g_e should be very close to -2 (Dirac value)
        assert!((ELECTRON_G_FACTOR.value + 2.0).abs() < 0.003);
    }

    #[test]
    fn test_compton_wavelength_derived() {
        // lambda_C = h / (m_e * c)
        let expected = PLANCK_CONSTANT.value / (ELECTRON_MASS.value * SPEED_OF_LIGHT.value);
        let relative_error = ((COMPTON_WAVELENGTH.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Compton wavelength derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_bohr_magneton_derived() {
        // mu_B = e * hbar / (2 * m_e)
        let expected =
            ELEMENTARY_CHARGE.value * REDUCED_PLANCK_CONSTANT.value / (2.0 * ELECTRON_MASS.value);
        let relative_error = ((BOHR_MAGNETON.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Bohr magneton derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_nuclear_magneton_ratio() {
        // mu_N / mu_B should be approximately m_e / m_p
        let magneton_ratio = NUCLEAR_MAGNETON.value / BOHR_MAGNETON.value;
        let mass_ratio =
            ELECTRON_MASS.value / crate::new_modules::constants::fundamental::PROTON_MASS.value;
        let relative_error = ((magneton_ratio - mass_ratio) / mass_ratio).abs();
        assert!(
            relative_error < REL_TOL,
            "Nuclear/Bohr magneton ratio error: {}",
            relative_error
        );
    }

    #[test]
    fn test_proton_magnetic_moment_in_nuclear_magnetons() {
        // mu_p / mu_N should be approximately 2.793
        let ratio = PROTON_MAGNETIC_MOMENT.value / NUCLEAR_MAGNETON.value;
        assert!(
            (ratio - 2.793).abs() < 0.001,
            "mu_p/mu_N = {} (expected ~2.793)",
            ratio
        );
    }

    #[test]
    fn test_bohr_radius_derived() {
        // a_0 = hbar / (m_e * c * alpha)
        let expected = REDUCED_PLANCK_CONSTANT.value
            / (ELECTRON_MASS.value * SPEED_OF_LIGHT.value * FINE_STRUCTURE_CONSTANT.value);
        let relative_error = ((BOHR_RADIUS.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Bohr radius derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_hartree_energy_derived() {
        // E_h = alpha^2 * m_e * c^2
        let alpha = FINE_STRUCTURE_CONSTANT.value;
        let expected =
            alpha * alpha * ELECTRON_MASS.value * SPEED_OF_LIGHT.value * SPEED_OF_LIGHT.value;
        let relative_error = ((HARTREE_ENERGY.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Hartree energy derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_all_atomic_constants_positive_except_g_factor() {
        assert!(BOHR_RADIUS.value > 0.0);
        assert!(RYDBERG_CONSTANT.value > 0.0);
        assert!(HARTREE_ENERGY.value > 0.0);
        assert!(CLASSICAL_ELECTRON_RADIUS.value > 0.0);
        assert!(COMPTON_WAVELENGTH.value > 0.0);
        assert!(BOHR_MAGNETON.value > 0.0);
        assert!(NUCLEAR_MAGNETON.value > 0.0);
        assert!(PROTON_MAGNETIC_MOMENT.value > 0.0);
        // g_e is negative
        assert!(ELECTRON_G_FACTOR.value < 0.0);
    }

    #[test]
    fn test_all_atomic_uncertainties_positive() {
        let constants = [
            &BOHR_RADIUS,
            &RYDBERG_CONSTANT,
            &HARTREE_ENERGY,
            &CLASSICAL_ELECTRON_RADIUS,
            &COMPTON_WAVELENGTH,
            &BOHR_MAGNETON,
            &NUCLEAR_MAGNETON,
            &PROTON_MAGNETIC_MOMENT,
            &ELECTRON_G_FACTOR,
        ];
        for c in &constants {
            assert!(
                c.uncertainty > 0.0,
                "{} should have positive uncertainty",
                c.name
            );
        }
    }

    #[test]
    fn test_rydberg_derived() {
        // R_inf = alpha^2 * m_e * c / (2 * h)
        let alpha = FINE_STRUCTURE_CONSTANT.value;
        let expected = alpha * alpha * ELECTRON_MASS.value * SPEED_OF_LIGHT.value
            / (2.0 * PLANCK_CONSTANT.value);
        let relative_error = ((RYDBERG_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < REL_TOL,
            "Rydberg constant derivation error: {}",
            relative_error
        );
    }
}
