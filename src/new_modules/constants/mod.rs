//! Physical Constants Module for NumRS2
//!
//! A comprehensive, NIST-compliant physical constants library providing
//! fundamental, atomic, nuclear, and electromagnetic constants based on
//! the CODATA 2022 internationally recommended values.
//!
//! # Module Organization
//!
//! - [`fundamental`]: Fundamental physical constants (speed of light, Planck, Boltzmann, etc.)
//! - [`atomic`]: Atomic and nuclear constants (Bohr radius, Rydberg, magneton, etc.)
//! - [`electromagnetic`]: Electromagnetic constants (permittivity, permeability, flux quantum, etc.)
//! - [`units`]: Unit conversion constants and functions (SI prefixes, energy, temperature, length)
//!
//! # Usage
//!
//! ```rust
//! use numrs2::new_modules::constants::{SPEED_OF_LIGHT, PLANCK_CONSTANT, BOLTZMANN_CONSTANT};
//!
//! // Access constant values
//! let c = SPEED_OF_LIGHT.value;
//! let h = PLANCK_CONSTANT.value;
//!
//! // Display full information
//! println!("{}", SPEED_OF_LIGHT);
//! // Output: Speed of light in vacuum (c) = 299792458 m/s (exact)
//! ```
//!
//! # Data Source
//!
//! All values are from CODATA 2022 (Committee on Data of the International
//! Science Council). Constants defined as exact in the 2019 SI redefinition
//! have zero uncertainty.
//!
//! # References
//!
//! - CODATA 2022: <https://physics.nist.gov/cuu/Constants/>
//! - 2019 SI Redefinition: <https://www.bipm.org/en/measurement-units/si-redefinition>

pub mod atomic;
pub mod electromagnetic;
pub mod fundamental;
pub mod units;

// Re-export commonly used fundamental constants at module level
pub use fundamental::{
    ATOMIC_MASS_UNIT, AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELECTRON_CHARGE_MASS_RATIO,
    ELECTRON_MASS, ELEMENTARY_CHARGE, FINE_STRUCTURE_CONSTANT, GAS_CONSTANT,
    GRAVITATIONAL_CONSTANT, NEUTRON_MASS, PLANCK_CONSTANT, PROTON_MASS, REDUCED_PLANCK_CONSTANT,
    SPEED_OF_LIGHT, STEFAN_BOLTZMANN_CONSTANT,
};

// Re-export commonly used atomic constants
pub use atomic::{
    BOHR_MAGNETON, BOHR_RADIUS, CLASSICAL_ELECTRON_RADIUS, COMPTON_WAVELENGTH, ELECTRON_G_FACTOR,
    HARTREE_ENERGY, NUCLEAR_MAGNETON, PROTON_MAGNETIC_MOMENT, RYDBERG_CONSTANT,
};

// Re-export commonly used electromagnetic constants
pub use electromagnetic::{
    CONDUCTANCE_QUANTUM, COULOMB_CONSTANT, IMPEDANCE_OF_FREE_SPACE, JOSEPHSON_CONSTANT,
    MAGNETIC_FLUX_QUANTUM, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY, VON_KLITZING_CONSTANT,
};

use std::fmt;

/// A physical constant with its value, uncertainty, unit, symbol, and name.
///
/// This struct represents a NIST/CODATA physical constant with full metadata.
/// Constants that are exact (zero uncertainty) under the 2019 SI redefinition
/// are indicated by `uncertainty = 0.0`.
///
/// # Fields
///
/// * `value` - The numerical value of the constant in SI units
/// * `uncertainty` - The standard uncertainty (0.0 for exact constants)
/// * `unit` - The SI unit string (e.g., "m/s", "J s")
/// * `symbol` - The conventional symbol (e.g., "c", "h", "k_B")
/// * `name` - The full descriptive name of the constant
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalConstant {
    /// The numerical value of the constant in SI units
    pub value: f64,
    /// The standard uncertainty (0.0 for exact constants)
    pub uncertainty: f64,
    /// The SI unit string
    pub unit: &'static str,
    /// The conventional symbol
    pub symbol: &'static str,
    /// The full descriptive name
    pub name: &'static str,
}

impl PhysicalConstant {
    /// Returns the relative uncertainty of this constant.
    ///
    /// For exact constants (uncertainty = 0.0), this returns 0.0.
    /// For non-exact constants, this returns `uncertainty / |value|`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::new_modules::constants::GRAVITATIONAL_CONSTANT;
    /// let rel = GRAVITATIONAL_CONSTANT.relative_uncertainty();
    /// assert!(rel > 0.0); // G has nonzero uncertainty
    /// ```
    pub fn relative_uncertainty(&self) -> f64 {
        if self.value == 0.0 || self.uncertainty == 0.0 {
            0.0
        } else {
            self.uncertainty / self.value.abs()
        }
    }

    /// Returns `true` if this constant is exact (zero uncertainty).
    ///
    /// Under the 2019 SI redefinition, several fundamental constants
    /// (speed of light, Planck constant, elementary charge, Boltzmann
    /// constant, Avogadro constant) have exact defined values.
    pub fn is_exact(&self) -> bool {
        self.uncertainty == 0.0
    }
}

impl fmt::Display for PhysicalConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_exact() {
            write!(
                f,
                "{} ({}) = {} {} (exact)",
                self.name, self.symbol, self.value, self.unit
            )
        } else {
            write!(
                f,
                "{} ({}) = {} {} (uncertainty: {})",
                self.name, self.symbol, self.value, self.unit, self.uncertainty
            )
        }
    }
}

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_constant_display_exact() {
        let c = PhysicalConstant {
            value: 299_792_458.0,
            uncertainty: 0.0,
            unit: "m/s",
            symbol: "c",
            name: "Speed of light in vacuum",
        };
        let display = format!("{}", c);
        assert!(display.contains("exact"));
        assert!(display.contains("299792458"));
        assert!(display.contains("m/s"));
    }

    #[test]
    fn test_physical_constant_display_with_uncertainty() {
        let g = PhysicalConstant {
            value: 6.674_30e-11,
            uncertainty: 1.5e-15,
            unit: "m^3 kg^-1 s^-2",
            symbol: "G",
            name: "Newtonian constant of gravitation",
        };
        let display = format!("{}", g);
        assert!(display.contains("uncertainty"));
        assert!(!display.contains("exact"));
    }

    #[test]
    fn test_is_exact() {
        assert!(SPEED_OF_LIGHT.is_exact());
        assert!(PLANCK_CONSTANT.is_exact());
        assert!(ELEMENTARY_CHARGE.is_exact());
        assert!(BOLTZMANN_CONSTANT.is_exact());
        assert!(AVOGADRO_CONSTANT.is_exact());
        assert!(!GRAVITATIONAL_CONSTANT.is_exact());
        assert!(!FINE_STRUCTURE_CONSTANT.is_exact());
    }

    #[test]
    fn test_relative_uncertainty_exact() {
        assert_eq!(SPEED_OF_LIGHT.relative_uncertainty(), 0.0);
    }

    #[test]
    fn test_relative_uncertainty_nonexact() {
        let rel = GRAVITATIONAL_CONSTANT.relative_uncertainty();
        assert!(rel > 0.0);
        assert!(rel < 1.0e-4); // G is known to better than 0.01%
    }

    #[test]
    fn test_physical_constant_clone() {
        let c1 = SPEED_OF_LIGHT;
        let c2 = c1;
        assert_eq!(c1.value, c2.value);
        assert_eq!(c1.symbol, c2.symbol);
    }

    #[test]
    fn test_physical_constant_debug() {
        let debug_str = format!("{:?}", SPEED_OF_LIGHT);
        assert!(debug_str.contains("PhysicalConstant"));
        assert!(debug_str.contains("value"));
    }

    #[test]
    fn test_zero_value_relative_uncertainty() {
        let zero_const = PhysicalConstant {
            value: 0.0,
            uncertainty: 0.0,
            unit: "",
            symbol: "",
            name: "Zero constant",
        };
        assert_eq!(zero_const.relative_uncertainty(), 0.0);
    }

    #[test]
    fn test_reexports_fundamental() {
        // Verify that re-exported constants are accessible
        assert!(SPEED_OF_LIGHT.value > 0.0);
        assert!(PLANCK_CONSTANT.value > 0.0);
        assert!(BOLTZMANN_CONSTANT.value > 0.0);
        assert!(AVOGADRO_CONSTANT.value > 0.0);
        assert!(ELEMENTARY_CHARGE.value > 0.0);
    }

    #[test]
    fn test_reexports_atomic() {
        assert!(BOHR_RADIUS.value > 0.0);
        assert!(RYDBERG_CONSTANT.value > 0.0);
        assert!(HARTREE_ENERGY.value > 0.0);
    }

    #[test]
    fn test_reexports_electromagnetic() {
        assert!(VACUUM_PERMITTIVITY.value > 0.0);
        assert!(VACUUM_PERMEABILITY.value > 0.0);
        assert!(IMPEDANCE_OF_FREE_SPACE.value > 0.0);
    }

    #[test]
    fn test_physical_constant_partial_eq() {
        let c1 = PhysicalConstant {
            value: 1.0,
            uncertainty: 0.0,
            unit: "m",
            symbol: "x",
            name: "Test",
        };
        let c2 = PhysicalConstant {
            value: 1.0,
            uncertainty: 0.0,
            unit: "m",
            symbol: "x",
            name: "Test",
        };
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_physical_constant_copy() {
        let c1 = PLANCK_CONSTANT;
        let c2 = c1;
        // Both should be usable after copy
        assert_eq!(c1.value, c2.value);
        assert_eq!(c1.uncertainty, c2.uncertainty);
    }

    #[test]
    fn test_unit_strings_not_empty() {
        assert!(!SPEED_OF_LIGHT.unit.is_empty());
        assert!(!PLANCK_CONSTANT.unit.is_empty());
        assert!(!GRAVITATIONAL_CONSTANT.unit.is_empty());
        assert!(!BOHR_RADIUS.unit.is_empty());
        assert!(!VACUUM_PERMITTIVITY.unit.is_empty());
    }

    #[test]
    fn test_symbol_strings_not_empty() {
        assert!(!SPEED_OF_LIGHT.symbol.is_empty());
        assert!(!PLANCK_CONSTANT.symbol.is_empty());
        assert!(!ELEMENTARY_CHARGE.symbol.is_empty());
    }
}
