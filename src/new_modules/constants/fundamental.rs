//! Fundamental Physical Constants (CODATA 2022)
//!
//! This module provides the fundamental physical constants as recommended by
//! the Committee on Data of the International Science Council (CODATA) 2022.
//!
//! Constants that are exact under the 2019 SI redefinition have zero uncertainty.
//! These include the speed of light, Planck constant, elementary charge,
//! Boltzmann constant, and Avogadro constant.
//!
//! # References
//!
//! - CODATA 2022: <https://physics.nist.gov/cuu/Constants/>
//! - E. Tiesinga, P.J. Mohr, D.B. Newell, and B.N. Taylor,
//!   "CODATA recommended values of the fundamental physical constants: 2022",
//!   Rev. Mod. Phys. (2024).

use super::PhysicalConstant;

/// Speed of light in vacuum (c)
///
/// Defined value: 299,792,458 m/s (exact under SI 2019)
///
/// The speed of light in vacuum is a fundamental constant of nature and
/// one of the seven defining constants of the International System of Units.
/// It serves as the basis for the definition of the metre.
pub const SPEED_OF_LIGHT: PhysicalConstant = PhysicalConstant {
    value: 299_792_458.0,
    uncertainty: 0.0,
    unit: "m/s",
    symbol: "c",
    name: "Speed of light in vacuum",
};

/// Planck constant (h)
///
/// Defined value: 6.626_070_15 x 10^-34 J s (exact under SI 2019)
///
/// The Planck constant relates the energy of a photon to its frequency
/// (E = hf). It is one of the seven defining constants of the SI and
/// serves as the basis for the definition of the kilogram.
pub const PLANCK_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 6.626_070_15e-34,
    uncertainty: 0.0,
    unit: "J s",
    symbol: "h",
    name: "Planck constant",
};

/// Reduced Planck constant (hbar = h / 2pi)
///
/// Value: 1.054_571_817... x 10^-34 J s (exact, derived from h)
///
/// Also known as the Dirac constant, hbar appears naturally in quantum
/// mechanics, particularly in the Schrodinger equation and angular momentum
/// quantization. It equals h/(2*pi).
pub const REDUCED_PLANCK_CONSTANT: PhysicalConstant = PhysicalConstant {
    // h / (2 * pi) = 6.626_070_15e-34 / 6.283_185_307_179_586
    // = 1.054_571_817_646_156_4e-34
    value: 1.054_571_817_646_156_4e-34,
    uncertainty: 0.0,
    unit: "J s",
    symbol: "\u{0127}", // Unicode ħ
    name: "Reduced Planck constant",
};

/// Elementary charge (e)
///
/// Defined value: 1.602_176_634 x 10^-19 C (exact under SI 2019)
///
/// The elementary charge is the electric charge carried by a single proton
/// (or the magnitude of charge carried by a single electron). It is one of
/// the seven defining constants of the SI.
pub const ELEMENTARY_CHARGE: PhysicalConstant = PhysicalConstant {
    value: 1.602_176_634e-19,
    uncertainty: 0.0,
    unit: "C",
    symbol: "e",
    name: "Elementary charge",
};

/// Boltzmann constant (k_B)
///
/// Defined value: 1.380_649 x 10^-23 J/K (exact under SI 2019)
///
/// The Boltzmann constant relates the average kinetic energy of particles
/// in a gas to the temperature of the gas. It is one of the seven defining
/// constants of the SI and serves as the basis for the definition of the kelvin.
pub const BOLTZMANN_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 1.380_649e-23,
    uncertainty: 0.0,
    unit: "J/K",
    symbol: "k_B",
    name: "Boltzmann constant",
};

/// Avogadro constant (N_A)
///
/// Defined value: 6.022_140_76 x 10^23 mol^-1 (exact under SI 2019)
///
/// The Avogadro constant is the number of constituent particles per mole
/// of substance. It is one of the seven defining constants of the SI and
/// serves as the basis for the definition of the mole.
pub const AVOGADRO_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 6.022_140_76e23,
    uncertainty: 0.0,
    unit: "mol^-1",
    symbol: "N_A",
    name: "Avogadro constant",
};

/// Newtonian constant of gravitation (G)
///
/// Value: 6.674_30 x 10^-11 m^3 kg^-1 s^-2
/// Uncertainty: 1.5 x 10^-15 m^3 kg^-1 s^-2
///
/// The gravitational constant appears in Newton's law of universal
/// gravitation and Einstein's general theory of relativity. It is one
/// of the least precisely known fundamental constants due to the
/// weakness of gravity and difficulties in measurement.
pub const GRAVITATIONAL_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 6.674_30e-11,
    uncertainty: 1.5e-15,
    unit: "m^3 kg^-1 s^-2",
    symbol: "G",
    name: "Newtonian constant of gravitation",
};

/// Fine-structure constant (alpha)
///
/// Value: 7.297_352_564_3 x 10^-3
/// Uncertainty: 1.1 x 10^-12
///
/// The fine-structure constant is a dimensionless quantity that characterizes
/// the strength of the electromagnetic interaction between elementary charged
/// particles. It is defined as alpha = e^2 / (4*pi*epsilon_0*hbar*c).
/// Its approximate value of 1/137 is one of the most famous numbers in physics.
pub const FINE_STRUCTURE_CONSTANT: PhysicalConstant = PhysicalConstant {
    value: 7.297_352_564_3e-3,
    uncertainty: 1.1e-12,
    unit: "",
    symbol: "\u{03B1}", // Unicode α
    name: "Fine-structure constant",
};

/// Electron mass (m_e)
///
/// Value: 9.109_383_713_9 x 10^-31 kg
/// Uncertainty: 2.8 x 10^-40 kg
///
/// The rest mass of the electron, a fundamental lepton. It is the lightest
/// charged particle and plays a central role in atomic structure and chemistry.
pub const ELECTRON_MASS: PhysicalConstant = PhysicalConstant {
    value: 9.109_383_713_9e-31,
    uncertainty: 2.8e-40,
    unit: "kg",
    symbol: "m_e",
    name: "Electron mass",
};

/// Proton mass (m_p)
///
/// Value: 1.672_621_925_95 x 10^-27 kg
/// Uncertainty: 5.2 x 10^-37 kg
///
/// The rest mass of the proton, one of the two nucleons (along with the neutron).
/// The proton is approximately 1836 times heavier than the electron.
pub const PROTON_MASS: PhysicalConstant = PhysicalConstant {
    value: 1.672_621_925_95e-27,
    uncertainty: 5.2e-37,
    unit: "kg",
    symbol: "m_p",
    name: "Proton mass",
};

/// Neutron mass (m_n)
///
/// Value: 1.674_927_500_56 x 10^-27 kg
/// Uncertainty: 8.5 x 10^-37 kg
///
/// The rest mass of the neutron, a neutral nucleon. The neutron is slightly
/// heavier than the proton, which is crucial for nuclear stability and
/// beta decay processes.
pub const NEUTRON_MASS: PhysicalConstant = PhysicalConstant {
    value: 1.674_927_500_56e-27,
    uncertainty: 8.5e-37,
    unit: "kg",
    symbol: "m_n",
    name: "Neutron mass",
};

/// Electron charge-to-mass ratio (e/m_e)
///
/// Value: -1.758_820_010_76 x 10^11 C/kg
/// Uncertainty: 5.3 x 10^1 C/kg
///
/// The ratio of the electron's charge to its mass. This quantity is directly
/// measurable in cathode ray and cyclotron experiments. The negative sign
/// reflects the electron's negative charge.
pub const ELECTRON_CHARGE_MASS_RATIO: PhysicalConstant = PhysicalConstant {
    // e / m_e = 1.602_176_634e-19 / 9.109_383_713_9e-31
    // = 1.758_820_010_76e11 (negative because electron charge is negative)
    value: -1.758_820_010_76e11,
    uncertainty: 5.3e1,
    unit: "C/kg",
    symbol: "-e/m_e",
    name: "Electron charge-to-mass ratio",
};

/// Atomic mass unit (unified atomic mass unit, u or Da)
///
/// Value: 1.660_539_068_92 x 10^-27 kg
/// Uncertainty: 5.2 x 10^-37 kg
///
/// The unified atomic mass unit is defined as 1/12 of the mass of an
/// unbound neutral atom of carbon-12 in its nuclear and electronic ground
/// state. It is widely used in atomic and molecular physics.
pub const ATOMIC_MASS_UNIT: PhysicalConstant = PhysicalConstant {
    value: 1.660_539_068_92e-27,
    uncertainty: 5.2e-37,
    unit: "kg",
    symbol: "u",
    name: "Unified atomic mass unit",
};

/// Stefan-Boltzmann constant (sigma)
///
/// Value: 5.670_374_419... x 10^-8 W m^-2 K^-4 (exact, derived)
///
/// The Stefan-Boltzmann constant relates the total energy radiated per unit
/// surface area of a black body to the fourth power of its absolute temperature.
/// It is derived from other exact constants: sigma = 2*pi^5*k_B^4 / (15*h^3*c^2).
pub const STEFAN_BOLTZMANN_CONSTANT: PhysicalConstant = PhysicalConstant {
    // sigma = 2 * pi^5 * k_B^4 / (15 * h^3 * c^2)
    // = 5.670_374_419_184_429_5e-8
    value: 5.670_374_419_184_429_5e-8,
    uncertainty: 0.0,
    unit: "W m^-2 K^-4",
    symbol: "\u{03C3}", // Unicode σ
    name: "Stefan-Boltzmann constant",
};

/// Molar gas constant (R)
///
/// Value: 8.314_462_618_153_24 J mol^-1 K^-1 (exact, derived)
///
/// The molar gas constant is the product of Avogadro's constant and
/// Boltzmann's constant: R = N_A * k_B. It appears in the ideal gas law
/// (PV = nRT) and many thermodynamic equations.
pub const GAS_CONSTANT: PhysicalConstant = PhysicalConstant {
    // R = N_A * k_B = 6.022_140_76e23 * 1.380_649e-23
    // = 8.314_462_618_153_24
    value: 8.314_462_618_153_24,
    uncertainty: 0.0,
    unit: "J mol^-1 K^-1",
    symbol: "R",
    name: "Molar gas constant",
};

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_speed_of_light_value() {
        assert_eq!(SPEED_OF_LIGHT.value, 299_792_458.0);
        assert!(SPEED_OF_LIGHT.is_exact());
    }

    #[test]
    fn test_planck_constant_value() {
        let expected = 6.626_070_15e-34;
        assert!((PLANCK_CONSTANT.value - expected).abs() < 1e-42);
        assert!(PLANCK_CONSTANT.is_exact());
    }

    #[test]
    fn test_reduced_planck_constant_derived() {
        // hbar should equal h / (2*pi)
        let expected = PLANCK_CONSTANT.value / (2.0 * std::f64::consts::PI);
        let relative_error = ((REDUCED_PLANCK_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-15,
            "hbar derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_elementary_charge_value() {
        assert_eq!(ELEMENTARY_CHARGE.value, 1.602_176_634e-19);
        assert!(ELEMENTARY_CHARGE.is_exact());
    }

    #[test]
    fn test_boltzmann_constant_value() {
        assert_eq!(BOLTZMANN_CONSTANT.value, 1.380_649e-23);
        assert!(BOLTZMANN_CONSTANT.is_exact());
    }

    #[test]
    fn test_avogadro_constant_value() {
        assert_eq!(AVOGADRO_CONSTANT.value, 6.022_140_76e23);
        assert!(AVOGADRO_CONSTANT.is_exact());
    }

    #[test]
    fn test_gravitational_constant_value() {
        let expected = 6.674_30e-11;
        assert!((GRAVITATIONAL_CONSTANT.value - expected).abs() < 1e-15);
        assert!(!GRAVITATIONAL_CONSTANT.is_exact());
        assert!(GRAVITATIONAL_CONSTANT.uncertainty > 0.0);
    }

    #[test]
    fn test_fine_structure_constant_value() {
        let expected = 7.297_352_564_3e-3;
        assert!((FINE_STRUCTURE_CONSTANT.value - expected).abs() < 1e-12);
        // alpha should be approximately 1/137
        let inv_alpha = 1.0 / FINE_STRUCTURE_CONSTANT.value;
        assert!((inv_alpha - 137.036).abs() < 0.001);
    }

    #[test]
    fn test_electron_mass_value() {
        let expected = 9.109_383_713_9e-31;
        assert!((ELECTRON_MASS.value - expected).abs() < 1e-40);
    }

    #[test]
    fn test_proton_mass_value() {
        let expected = 1.672_621_925_95e-27;
        assert!((PROTON_MASS.value - expected).abs() < 1e-37);
    }

    #[test]
    fn test_neutron_mass_value() {
        let expected = 1.674_927_500_56e-27;
        assert!((NEUTRON_MASS.value - expected).abs() < 1e-37);
    }

    #[test]
    fn test_proton_heavier_than_electron() {
        // Proton should be ~1836 times heavier than electron
        let ratio = PROTON_MASS.value / ELECTRON_MASS.value;
        assert!((ratio - 1836.15).abs() < 0.1);
    }

    #[test]
    fn test_neutron_heavier_than_proton() {
        assert!(NEUTRON_MASS.value > PROTON_MASS.value);
        let diff = NEUTRON_MASS.value - PROTON_MASS.value;
        // Neutron-proton mass difference is about 2.3 MeV/c^2
        assert!(diff > 0.0);
        assert!(diff < 1e-27);
    }

    #[test]
    fn test_electron_charge_mass_ratio() {
        // |e/m_e| should match e/m_e computed from components
        let expected = ELEMENTARY_CHARGE.value / ELECTRON_MASS.value;
        let actual = ELECTRON_CHARGE_MASS_RATIO.value.abs();
        let relative_error = ((actual - expected) / expected).abs();
        assert!(relative_error < EPSILON, "e/m_e error: {}", relative_error);
    }

    #[test]
    fn test_gas_constant_derived() {
        // R = N_A * k_B
        let expected = AVOGADRO_CONSTANT.value * BOLTZMANN_CONSTANT.value;
        let relative_error = ((GAS_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-12,
            "R derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_stefan_boltzmann_derived() {
        // sigma = 2 * pi^5 * k_B^4 / (15 * h^3 * c^2)
        let pi = std::f64::consts::PI;
        let k = BOLTZMANN_CONSTANT.value;
        let h = PLANCK_CONSTANT.value;
        let c = SPEED_OF_LIGHT.value;
        let expected = 2.0 * pi.powi(5) * k.powi(4) / (15.0 * h.powi(3) * c.powi(2));
        let relative_error = ((STEFAN_BOLTZMANN_CONSTANT.value - expected) / expected).abs();
        assert!(
            relative_error < 1e-10,
            "Stefan-Boltzmann derivation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_atomic_mass_unit_value() {
        let expected = 1.660_539_068_92e-27;
        assert!((ATOMIC_MASS_UNIT.value - expected).abs() < 1e-37);
    }

    #[test]
    fn test_all_units_nonempty() {
        // All constants except dimensionless ones should have non-empty units
        assert!(!SPEED_OF_LIGHT.unit.is_empty());
        assert!(!PLANCK_CONSTANT.unit.is_empty());
        assert!(!ELEMENTARY_CHARGE.unit.is_empty());
        assert!(!BOLTZMANN_CONSTANT.unit.is_empty());
        assert!(!AVOGADRO_CONSTANT.unit.is_empty());
        assert!(!GRAVITATIONAL_CONSTANT.unit.is_empty());
        assert!(!ELECTRON_MASS.unit.is_empty());
        assert!(!GAS_CONSTANT.unit.is_empty());
        // Fine-structure constant is dimensionless
        assert!(FINE_STRUCTURE_CONSTANT.unit.is_empty());
    }

    #[test]
    fn test_all_names_nonempty() {
        assert!(!SPEED_OF_LIGHT.name.is_empty());
        assert!(!PLANCK_CONSTANT.name.is_empty());
        assert!(!REDUCED_PLANCK_CONSTANT.name.is_empty());
        assert!(!ELEMENTARY_CHARGE.name.is_empty());
        assert!(!BOLTZMANN_CONSTANT.name.is_empty());
        assert!(!AVOGADRO_CONSTANT.name.is_empty());
        assert!(!GRAVITATIONAL_CONSTANT.name.is_empty());
        assert!(!FINE_STRUCTURE_CONSTANT.name.is_empty());
        assert!(!ELECTRON_MASS.name.is_empty());
        assert!(!PROTON_MASS.name.is_empty());
        assert!(!NEUTRON_MASS.name.is_empty());
        assert!(!ELECTRON_CHARGE_MASS_RATIO.name.is_empty());
        assert!(!ATOMIC_MASS_UNIT.name.is_empty());
        assert!(!STEFAN_BOLTZMANN_CONSTANT.name.is_empty());
        assert!(!GAS_CONSTANT.name.is_empty());
    }

    #[test]
    fn test_exact_constants_zero_uncertainty() {
        let exact_constants = [
            &SPEED_OF_LIGHT,
            &PLANCK_CONSTANT,
            &REDUCED_PLANCK_CONSTANT,
            &ELEMENTARY_CHARGE,
            &BOLTZMANN_CONSTANT,
            &AVOGADRO_CONSTANT,
            &STEFAN_BOLTZMANN_CONSTANT,
            &GAS_CONSTANT,
        ];
        for c in &exact_constants {
            assert_eq!(
                c.uncertainty, 0.0,
                "{} should be exact but has uncertainty {}",
                c.name, c.uncertainty
            );
        }
    }

    #[test]
    fn test_nonexact_constants_positive_uncertainty() {
        let nonexact_constants = [
            &GRAVITATIONAL_CONSTANT,
            &FINE_STRUCTURE_CONSTANT,
            &ELECTRON_MASS,
            &PROTON_MASS,
            &NEUTRON_MASS,
            &ATOMIC_MASS_UNIT,
        ];
        for c in &nonexact_constants {
            assert!(
                c.uncertainty > 0.0,
                "{} should have positive uncertainty",
                c.name
            );
        }
    }
}
