//! Unit Conversion Constants and Functions
//!
//! This module provides SI prefix multipliers, energy conversion constants,
//! temperature conversion functions, length conversion constants, and
//! time conversion constants.
//!
//! # SI Prefixes
//!
//! All 24 SI prefixes from yocto (10^-24) to yotta (10^24) are provided.
//!
//! # Energy Conversions
//!
//! Conversions between electronvolts, joules, kilocalories per mole,
//! kilojoules per mole, Hartree energy units, and Rydberg energy units.
//!
//! # Temperature Conversions
//!
//! Bidirectional conversions between kelvin, Celsius, Fahrenheit, and Rankine.
//!
//! # Length Conversions
//!
//! Conversions between metres, angstroms, Bohr radii, and femtometres.
//!
//! # References
//!
//! - SI Brochure (9th edition, 2019): <https://www.bipm.org/en/publications/si-brochure>
//! - CODATA 2022: <https://physics.nist.gov/cuu/Constants/>

use super::PhysicalConstant;

// ============================================================================
// SI Prefixes
// ============================================================================

/// SI prefix yocto (y): 10^-24
pub const YOCTO: f64 = 1e-24;

/// SI prefix zepto (z): 10^-21
pub const ZEPTO: f64 = 1e-21;

/// SI prefix atto (a): 10^-18
pub const ATTO: f64 = 1e-18;

/// SI prefix femto (f): 10^-15
pub const FEMTO: f64 = 1e-15;

/// SI prefix pico (p): 10^-12
pub const PICO: f64 = 1e-12;

/// SI prefix nano (n): 10^-9
pub const NANO: f64 = 1e-9;

/// SI prefix micro (mu): 10^-6
pub const MICRO: f64 = 1e-6;

/// SI prefix milli (m): 10^-3
pub const MILLI: f64 = 1e-3;

/// SI prefix centi (c): 10^-2
pub const CENTI: f64 = 1e-2;

/// SI prefix deci (d): 10^-1
pub const DECI: f64 = 1e-1;

/// SI prefix deca (da): 10^1
pub const DECA: f64 = 1e1;

/// SI prefix hecto (h): 10^2
pub const HECTO: f64 = 1e2;

/// SI prefix kilo (k): 10^3
pub const KILO: f64 = 1e3;

/// SI prefix mega (M): 10^6
pub const MEGA: f64 = 1e6;

/// SI prefix giga (G): 10^9
pub const GIGA: f64 = 1e9;

/// SI prefix tera (T): 10^12
pub const TERA: f64 = 1e12;

/// SI prefix peta (P): 10^15
pub const PETA: f64 = 1e15;

/// SI prefix exa (E): 10^18
pub const EXA: f64 = 1e18;

/// SI prefix zetta (Z): 10^21
pub const ZETTA: f64 = 1e21;

/// SI prefix yotta (Y): 10^24
pub const YOTTA: f64 = 1e24;

/// SI prefix ronto (r): 10^-27 (adopted in 2022)
pub const RONTO: f64 = 1e-27;

/// SI prefix quecto (q): 10^-30 (adopted in 2022)
pub const QUECTO: f64 = 1e-30;

/// SI prefix ronna (R): 10^27 (adopted in 2022)
pub const RONNA: f64 = 1e27;

/// SI prefix quetta (Q): 10^30 (adopted in 2022)
pub const QUETTA: f64 = 1e30;

// ============================================================================
// Energy Conversion Constants
// ============================================================================

/// Electronvolt to joule conversion factor
///
/// 1 eV = 1.602_176_634 x 10^-19 J (exact, from defined elementary charge)
pub const EV_TO_JOULE: PhysicalConstant = PhysicalConstant {
    value: 1.602_176_634e-19,
    uncertainty: 0.0,
    unit: "J/eV",
    symbol: "eV->J",
    name: "Electronvolt to joule",
};

/// Joule to electronvolt conversion factor
///
/// 1 J = 6.241_509_074... x 10^18 eV (exact, reciprocal of eV_to_J)
pub const JOULE_TO_EV: PhysicalConstant = PhysicalConstant {
    // 1 / 1.602_176_634e-19 = 6.241_509_074_460_763_4e18
    value: 6.241_509_074_460_763_4e18,
    uncertainty: 0.0,
    unit: "eV/J",
    symbol: "J->eV",
    name: "Joule to electronvolt",
};

/// Electronvolt to kilocalorie per mole conversion factor
///
/// 1 eV = 23.0605 kcal/mol (approximately)
///
/// Derived from: 1 eV * N_A / 4184
pub const EV_TO_KCAL_PER_MOL: PhysicalConstant = PhysicalConstant {
    // 1.602_176_634e-19 * 6.022_140_76e23 / 4184.0
    // = 23.060_547_830...
    value: 23.060_547_830_619_15,
    uncertainty: 0.0,
    unit: "kcal mol^-1 eV^-1",
    symbol: "eV->kcal/mol",
    name: "Electronvolt to kilocalorie per mole",
};

/// Electronvolt to kilojoule per mole conversion factor
///
/// 1 eV = 96.4853 kJ/mol (approximately)
///
/// Derived from: 1 eV * N_A / 1000
pub const EV_TO_KJ_PER_MOL: PhysicalConstant = PhysicalConstant {
    // 1.602_176_634e-19 * 6.022_140_76e23 / 1000.0
    // = 96.485_332_12...
    value: 96.485_332_123_310_03,
    uncertainty: 0.0,
    unit: "kJ mol^-1 eV^-1",
    symbol: "eV->kJ/mol",
    name: "Electronvolt to kilojoule per mole",
};

/// Hartree to electronvolt conversion factor
///
/// 1 E_h = 27.211_386_245_988 eV
/// Uncertainty: 5.3 x 10^-12 eV
///
/// Derived from: E_h / e = 4.359_744_722_206_0e-18 / 1.602_176_634e-19
pub const HARTREE_TO_EV: PhysicalConstant = PhysicalConstant {
    value: 27.211_386_245_988,
    uncertainty: 5.3e-12,
    unit: "eV/E_h",
    symbol: "E_h->eV",
    name: "Hartree to electronvolt",
};

/// Rydberg to electronvolt conversion factor
///
/// 1 Ry = 13.605_693_122_994 eV (= E_h / 2e)
/// Uncertainty: 2.6 x 10^-12 eV
///
/// One Rydberg is exactly half a Hartree.
pub const RYDBERG_TO_EV: PhysicalConstant = PhysicalConstant {
    value: 13.605_693_122_994,
    uncertainty: 2.6e-12,
    unit: "eV/Ry",
    symbol: "Ry->eV",
    name: "Rydberg to electronvolt",
};

// ============================================================================
// Length Conversion Constants
// ============================================================================

/// Angstrom to metre conversion factor
///
/// 1 angstrom = 10^-10 m (exact by definition)
pub const ANGSTROM_TO_METRE: PhysicalConstant = PhysicalConstant {
    value: 1e-10,
    uncertainty: 0.0,
    unit: "m",
    symbol: "A->m",
    name: "Angstrom to metre",
};

/// Metre to angstrom conversion factor
///
/// 1 m = 10^10 angstrom (exact by definition)
pub const METRE_TO_ANGSTROM: PhysicalConstant = PhysicalConstant {
    value: 1e10,
    uncertainty: 0.0,
    unit: "angstrom",
    symbol: "m->A",
    name: "Metre to angstrom",
};

/// Bohr radius to metre conversion factor
///
/// 1 a_0 = 5.291_772_105_44 x 10^-11 m
/// Uncertainty: 8.2 x 10^-21 m
pub const BOHR_TO_METRE: PhysicalConstant = PhysicalConstant {
    value: 5.291_772_105_44e-11,
    uncertainty: 8.2e-21,
    unit: "m/a_0",
    symbol: "a_0->m",
    name: "Bohr radius to metre",
};

/// Metre to Bohr radius conversion factor
///
/// 1 m = 1.889_726_124_6... x 10^10 a_0
/// Uncertainty: derived from Bohr radius uncertainty
pub const METRE_TO_BOHR: PhysicalConstant = PhysicalConstant {
    // 1 / 5.291_772_105_44e-11 = 1.889_726_124_626_e10
    value: 1.889_726_124_626e10,
    uncertainty: 2.9e0,
    unit: "a_0/m",
    symbol: "m->a_0",
    name: "Metre to Bohr radius",
};

/// Femtometre to metre conversion factor
///
/// 1 fm = 10^-15 m (exact by definition)
///
/// The femtometre (also called the fermi) is commonly used in nuclear
/// physics to express nuclear dimensions.
pub const FEMTOMETRE_TO_METRE: PhysicalConstant = PhysicalConstant {
    value: 1e-15,
    uncertainty: 0.0,
    unit: "m/fm",
    symbol: "fm->m",
    name: "Femtometre to metre",
};

/// Metre to femtometre conversion factor
///
/// 1 m = 10^15 fm (exact by definition)
pub const METRE_TO_FEMTOMETRE: PhysicalConstant = PhysicalConstant {
    value: 1e15,
    uncertainty: 0.0,
    unit: "fm/m",
    symbol: "m->fm",
    name: "Metre to femtometre",
};

// ============================================================================
// Time Conversion Constants
// ============================================================================

/// Atomic unit of time to second conversion factor
///
/// 1 atomic time unit = hbar / E_h = 2.418_884_326_585_7 x 10^-17 s
/// Uncertainty: 4.7 x 10^-29 s
///
/// The atomic unit of time is the natural timescale in atomic physics.
/// It is approximately 24.2 attoseconds.
pub const ATOMIC_TIME_UNIT_TO_SECOND: PhysicalConstant = PhysicalConstant {
    value: 2.418_884_326_585_7e-17,
    uncertainty: 4.7e-29,
    unit: "s",
    symbol: "a.u.t->s",
    name: "Atomic unit of time to second",
};

/// Second to atomic unit of time conversion factor
///
/// 1 s = 4.134_137_333_656... x 10^16 atomic time units
/// Uncertainty: derived from atomic time unit uncertainty
pub const SECOND_TO_ATOMIC_TIME_UNIT: PhysicalConstant = PhysicalConstant {
    // 1 / 2.418_884_326_585_7e-17 = 4.134_137_333_656e16
    value: 4.134_137_333_656e16,
    uncertainty: 8.0e4,
    unit: "a.u.t/s",
    symbol: "s->a.u.t",
    name: "Second to atomic unit of time",
};

// ============================================================================
// Conversion Functions
// ============================================================================

/// Convert energy from electronvolts to joules.
///
/// # Arguments
///
/// * `ev` - Energy in electronvolts
///
/// # Returns
///
/// Energy in joules
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::ev_to_joules;
/// let j = ev_to_joules(1.0);
/// assert!((j - 1.602_176_634e-19).abs() < 1e-28);
/// ```
pub fn ev_to_joules(ev: f64) -> f64 {
    ev * EV_TO_JOULE.value
}

/// Convert energy from joules to electronvolts.
///
/// # Arguments
///
/// * `j` - Energy in joules
///
/// # Returns
///
/// Energy in electronvolts
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::joules_to_ev;
/// let ev = joules_to_ev(1.602_176_634e-19);
/// assert!((ev - 1.0).abs() < 1e-10);
/// ```
pub fn joules_to_ev(j: f64) -> f64 {
    j * JOULE_TO_EV.value
}

/// Convert energy from electronvolts to kilocalories per mole.
///
/// # Arguments
///
/// * `ev` - Energy in electronvolts
///
/// # Returns
///
/// Energy in kcal/mol
pub fn ev_to_kcal_per_mol(ev: f64) -> f64 {
    ev * EV_TO_KCAL_PER_MOL.value
}

/// Convert energy from kilocalories per mole to electronvolts.
///
/// # Arguments
///
/// * `kcal` - Energy in kcal/mol
///
/// # Returns
///
/// Energy in electronvolts
pub fn kcal_per_mol_to_ev(kcal: f64) -> f64 {
    kcal / EV_TO_KCAL_PER_MOL.value
}

/// Convert energy from electronvolts to kilojoules per mole.
///
/// # Arguments
///
/// * `ev` - Energy in electronvolts
///
/// # Returns
///
/// Energy in kJ/mol
pub fn ev_to_kj_per_mol(ev: f64) -> f64 {
    ev * EV_TO_KJ_PER_MOL.value
}

/// Convert energy from kilojoules per mole to electronvolts.
///
/// # Arguments
///
/// * `kj` - Energy in kJ/mol
///
/// # Returns
///
/// Energy in electronvolts
pub fn kj_per_mol_to_ev(kj: f64) -> f64 {
    kj / EV_TO_KJ_PER_MOL.value
}

/// Convert energy from Hartree to electronvolts.
///
/// # Arguments
///
/// * `hartree` - Energy in Hartree units
///
/// # Returns
///
/// Energy in electronvolts
pub fn hartree_to_ev(hartree: f64) -> f64 {
    hartree * HARTREE_TO_EV.value
}

/// Convert energy from electronvolts to Hartree.
///
/// # Arguments
///
/// * `ev` - Energy in electronvolts
///
/// # Returns
///
/// Energy in Hartree units
pub fn ev_to_hartree(ev: f64) -> f64 {
    ev / HARTREE_TO_EV.value
}

/// Convert energy from Rydberg to electronvolts.
///
/// # Arguments
///
/// * `ry` - Energy in Rydberg units
///
/// # Returns
///
/// Energy in electronvolts
pub fn rydberg_to_ev(ry: f64) -> f64 {
    ry * RYDBERG_TO_EV.value
}

/// Convert energy from electronvolts to Rydberg.
///
/// # Arguments
///
/// * `ev` - Energy in electronvolts
///
/// # Returns
///
/// Energy in Rydberg units
pub fn ev_to_rydberg(ev: f64) -> f64 {
    ev / RYDBERG_TO_EV.value
}

/// Convert temperature from kelvin to Celsius.
///
/// # Arguments
///
/// * `k` - Temperature in kelvin
///
/// # Returns
///
/// Temperature in degrees Celsius
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::kelvin_to_celsius;
/// let c = kelvin_to_celsius(373.15);
/// assert!((c - 100.0).abs() < 1e-10);
/// ```
pub fn kelvin_to_celsius(k: f64) -> f64 {
    k - 273.15
}

/// Convert temperature from Celsius to kelvin.
///
/// # Arguments
///
/// * `c` - Temperature in degrees Celsius
///
/// # Returns
///
/// Temperature in kelvin
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::celsius_to_kelvin;
/// let k = celsius_to_kelvin(0.0);
/// assert!((k - 273.15).abs() < 1e-10);
/// ```
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Convert temperature from kelvin to Fahrenheit.
///
/// # Arguments
///
/// * `k` - Temperature in kelvin
///
/// # Returns
///
/// Temperature in degrees Fahrenheit
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::kelvin_to_fahrenheit;
/// let f = kelvin_to_fahrenheit(373.15);
/// assert!((f - 212.0).abs() < 1e-6);
/// ```
pub fn kelvin_to_fahrenheit(k: f64) -> f64 {
    (k - 273.15) * 9.0 / 5.0 + 32.0
}

/// Convert temperature from Fahrenheit to kelvin.
///
/// # Arguments
///
/// * `f` - Temperature in degrees Fahrenheit
///
/// # Returns
///
/// Temperature in kelvin
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::fahrenheit_to_kelvin;
/// let k = fahrenheit_to_kelvin(32.0);
/// assert!((k - 273.15).abs() < 1e-10);
/// ```
pub fn fahrenheit_to_kelvin(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0 + 273.15
}

/// Convert temperature from kelvin to Rankine.
///
/// # Arguments
///
/// * `k` - Temperature in kelvin
///
/// # Returns
///
/// Temperature in Rankine
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::kelvin_to_rankine;
/// let r = kelvin_to_rankine(273.15);
/// assert!((r - 491.67).abs() < 1e-6);
/// ```
pub fn kelvin_to_rankine(k: f64) -> f64 {
    k * 9.0 / 5.0
}

/// Convert temperature from Rankine to kelvin.
///
/// # Arguments
///
/// * `r` - Temperature in Rankine
///
/// # Returns
///
/// Temperature in kelvin
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::constants::units::rankine_to_kelvin;
/// let k = rankine_to_kelvin(491.67);
/// assert!((k - 273.15).abs() < 1e-6);
/// ```
pub fn rankine_to_kelvin(r: f64) -> f64 {
    r * 5.0 / 9.0
}

/// Convert length from metres to angstroms.
///
/// # Arguments
///
/// * `m` - Length in metres
///
/// # Returns
///
/// Length in angstroms
pub fn metres_to_angstroms(m: f64) -> f64 {
    m * METRE_TO_ANGSTROM.value
}

/// Convert length from angstroms to metres.
///
/// # Arguments
///
/// * `a` - Length in angstroms
///
/// # Returns
///
/// Length in metres
pub fn angstroms_to_metres(a: f64) -> f64 {
    a * ANGSTROM_TO_METRE.value
}

/// Convert length from metres to Bohr radii.
///
/// # Arguments
///
/// * `m` - Length in metres
///
/// # Returns
///
/// Length in Bohr radii (atomic units of length)
pub fn metres_to_bohr(m: f64) -> f64 {
    m * METRE_TO_BOHR.value
}

/// Convert length from Bohr radii to metres.
///
/// # Arguments
///
/// * `bohr` - Length in Bohr radii
///
/// # Returns
///
/// Length in metres
pub fn bohr_to_metres(bohr: f64) -> f64 {
    bohr * BOHR_TO_METRE.value
}

/// Convert length from metres to femtometres.
///
/// # Arguments
///
/// * `m` - Length in metres
///
/// # Returns
///
/// Length in femtometres
pub fn metres_to_femtometres(m: f64) -> f64 {
    m * METRE_TO_FEMTOMETRE.value
}

/// Convert length from femtometres to metres.
///
/// # Arguments
///
/// * `fm` - Length in femtometres
///
/// # Returns
///
/// Length in metres
pub fn femtometres_to_metres(fm: f64) -> f64 {
    fm * FEMTOMETRE_TO_METRE.value
}

/// Convert time from seconds to atomic time units.
///
/// # Arguments
///
/// * `s` - Time in seconds
///
/// # Returns
///
/// Time in atomic time units (hbar / E_h)
pub fn seconds_to_atomic_time(s: f64) -> f64 {
    s * SECOND_TO_ATOMIC_TIME_UNIT.value
}

/// Convert time from atomic time units to seconds.
///
/// # Arguments
///
/// * `atu` - Time in atomic time units
///
/// # Returns
///
/// Time in seconds
pub fn atomic_time_to_seconds(atu: f64) -> f64 {
    atu * ATOMIC_TIME_UNIT_TO_SECOND.value
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // ---- SI Prefix Tests ----

    #[test]
    fn test_si_prefix_values() {
        assert_eq!(YOCTO, 1e-24);
        assert_eq!(ZEPTO, 1e-21);
        assert_eq!(ATTO, 1e-18);
        assert_eq!(FEMTO, 1e-15);
        assert_eq!(PICO, 1e-12);
        assert_eq!(NANO, 1e-9);
        assert_eq!(MICRO, 1e-6);
        assert_eq!(MILLI, 1e-3);
        assert_eq!(CENTI, 1e-2);
        assert_eq!(DECI, 1e-1);
    }

    #[test]
    fn test_si_prefix_large_values() {
        assert_eq!(DECA, 1e1);
        assert_eq!(HECTO, 1e2);
        assert_eq!(KILO, 1e3);
        assert_eq!(MEGA, 1e6);
        assert_eq!(GIGA, 1e9);
        assert_eq!(TERA, 1e12);
        assert_eq!(PETA, 1e15);
        assert_eq!(EXA, 1e18);
        assert_eq!(ZETTA, 1e21);
        assert_eq!(YOTTA, 1e24);
    }

    #[test]
    fn test_si_prefix_2022_additions() {
        assert_eq!(RONTO, 1e-27);
        assert_eq!(QUECTO, 1e-30);
        assert_eq!(RONNA, 1e27);
        assert_eq!(QUETTA, 1e30);
    }

    #[test]
    fn test_si_prefix_symmetry() {
        // Each small prefix should be the reciprocal of its large counterpart
        assert!((YOCTO * YOTTA - 1.0).abs() < EPSILON);
        assert!((ZEPTO * ZETTA - 1.0).abs() < EPSILON);
        assert!((ATTO * EXA - 1.0).abs() < EPSILON);
        assert!((FEMTO * PETA - 1.0).abs() < EPSILON);
        assert!((PICO * TERA - 1.0).abs() < EPSILON);
        assert!((NANO * GIGA - 1.0).abs() < EPSILON);
        assert!((MICRO * MEGA - 1.0).abs() < EPSILON);
        assert!((MILLI * KILO - 1.0).abs() < EPSILON);
    }

    // ---- Energy Conversion Tests ----

    #[test]
    fn test_ev_joule_roundtrip() {
        let original = 42.0;
        let joules = ev_to_joules(original);
        let back = joules_to_ev(joules);
        assert!((back - original).abs() < EPSILON * original.abs());
    }

    #[test]
    fn test_ev_to_joules_one_ev() {
        let j = ev_to_joules(1.0);
        assert!((j - 1.602_176_634e-19).abs() < 1e-28);
    }

    #[test]
    fn test_joules_to_ev_one_joule() {
        let ev = joules_to_ev(1.0);
        assert!((ev - 6.241_509_074_460_763_4e18).abs() < 1e8);
    }

    #[test]
    fn test_ev_kcal_roundtrip() {
        let original = 3.5;
        let kcal = ev_to_kcal_per_mol(original);
        let back = kcal_per_mol_to_ev(kcal);
        assert!((back - original).abs() < EPSILON * original.abs());
    }

    #[test]
    fn test_ev_kj_roundtrip() {
        let original = 7.8;
        let kj = ev_to_kj_per_mol(original);
        let back = kj_per_mol_to_ev(kj);
        assert!((back - original).abs() < EPSILON * original.abs());
    }

    #[test]
    fn test_hartree_ev_roundtrip() {
        let original = 2.0;
        let ev = hartree_to_ev(original);
        let back = ev_to_hartree(ev);
        assert!((back - original).abs() < EPSILON * original.abs());
    }

    #[test]
    fn test_rydberg_ev_roundtrip() {
        let original = 5.0;
        let ev = rydberg_to_ev(original);
        let back = ev_to_rydberg(ev);
        assert!((back - original).abs() < EPSILON * original.abs());
    }

    #[test]
    fn test_hartree_rydberg_relation() {
        // 1 Hartree = 2 Rydberg
        let hartree_ev = hartree_to_ev(1.0);
        let rydberg_ev = rydberg_to_ev(2.0);
        let relative_error = ((hartree_ev - rydberg_ev) / hartree_ev).abs();
        assert!(
            relative_error < 1e-10,
            "1 Hartree should equal 2 Rydberg: {} vs {}",
            hartree_ev,
            rydberg_ev
        );
    }

    // ---- Temperature Conversion Tests ----

    #[test]
    fn test_kelvin_celsius_boiling() {
        let c = kelvin_to_celsius(373.15);
        assert!((c - 100.0).abs() < EPSILON);
    }

    #[test]
    fn test_kelvin_celsius_freezing() {
        let c = kelvin_to_celsius(273.15);
        assert!(c.abs() < EPSILON);
    }

    #[test]
    fn test_celsius_kelvin_roundtrip() {
        let original = 25.0;
        let k = celsius_to_kelvin(original);
        let back = kelvin_to_celsius(k);
        assert!((back - original).abs() < EPSILON);
    }

    #[test]
    fn test_kelvin_fahrenheit_boiling() {
        let f = kelvin_to_fahrenheit(373.15);
        assert!((f - 212.0).abs() < 1e-6);
    }

    #[test]
    fn test_fahrenheit_kelvin_freezing() {
        let k = fahrenheit_to_kelvin(32.0);
        assert!((k - 273.15).abs() < EPSILON);
    }

    #[test]
    fn test_fahrenheit_kelvin_roundtrip() {
        let original = 98.6;
        let k = fahrenheit_to_kelvin(original);
        let back = kelvin_to_fahrenheit(k);
        assert!((back - original).abs() < 1e-9);
    }

    #[test]
    fn test_kelvin_rankine_freezing() {
        let r = kelvin_to_rankine(273.15);
        assert!((r - 491.67).abs() < 1e-6);
    }

    #[test]
    fn test_rankine_kelvin_roundtrip() {
        let original = 500.0;
        let k = rankine_to_kelvin(original);
        let back = kelvin_to_rankine(k);
        assert!((back - original).abs() < EPSILON);
    }

    #[test]
    fn test_absolute_zero_all_scales() {
        let k = 0.0;
        let c = kelvin_to_celsius(k);
        let f = kelvin_to_fahrenheit(k);
        let r = kelvin_to_rankine(k);
        assert!((c - (-273.15)).abs() < EPSILON);
        assert!((f - (-459.67)).abs() < 1e-6);
        assert!(r.abs() < EPSILON);
    }

    // ---- Length Conversion Tests ----

    #[test]
    fn test_metre_angstrom_roundtrip() {
        let original = 1.5e-10;
        let angstroms = metres_to_angstroms(original);
        let back = angstroms_to_metres(angstroms);
        assert!((back - original).abs() < 1e-25);
    }

    #[test]
    fn test_one_angstrom_in_metres() {
        let m = angstroms_to_metres(1.0);
        assert!((m - 1e-10).abs() < 1e-25);
    }

    #[test]
    fn test_metre_bohr_roundtrip() {
        let original = 1e-10;
        let bohr = metres_to_bohr(original);
        let back = bohr_to_metres(bohr);
        let relative_error = ((back - original) / original).abs();
        assert!(relative_error < 1e-5);
    }

    #[test]
    fn test_one_bohr_in_metres() {
        let m = bohr_to_metres(1.0);
        assert!((m - 5.291_772_105_44e-11).abs() < 1e-20);
    }

    #[test]
    fn test_metre_femtometre_roundtrip() {
        let original = 1e-15;
        let fm = metres_to_femtometres(original);
        assert!((fm - 1.0).abs() < EPSILON);
        let back = femtometres_to_metres(fm);
        assert!((back - original).abs() < 1e-30);
    }

    // ---- Time Conversion Tests ----

    #[test]
    fn test_atomic_time_seconds_roundtrip() {
        let original = 1e-15; // 1 femtosecond
        let atu = seconds_to_atomic_time(original);
        let back = atomic_time_to_seconds(atu);
        let relative_error = ((back - original) / original).abs();
        assert!(relative_error < 1e-5);
    }

    #[test]
    fn test_one_atomic_time_unit() {
        let s = atomic_time_to_seconds(1.0);
        assert!((s - 2.418_884_326_585_7e-17).abs() < 1e-28);
    }

    // ---- Conversion Constant Tests ----

    #[test]
    fn test_ev_joule_reciprocal() {
        // EV_TO_JOULE and JOULE_TO_EV should be reciprocals
        let product = EV_TO_JOULE.value * JOULE_TO_EV.value;
        assert!((product - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angstrom_metre_reciprocal() {
        let product = ANGSTROM_TO_METRE.value * METRE_TO_ANGSTROM.value;
        assert!((product - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_femtometre_metre_reciprocal() {
        let product = FEMTOMETRE_TO_METRE.value * METRE_TO_FEMTOMETRE.value;
        assert!((product - 1.0).abs() < EPSILON);
    }
}
