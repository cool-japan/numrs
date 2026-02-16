//! # Wavelet Families
//!
//! This module provides implementations of various wavelet families used in
//! discrete wavelet transforms. Each wavelet family has specific properties
//! that make it suitable for different applications.
//!
//! ## Wavelet Families
//!
//! - **Haar**: Simplest wavelet, discontinuous, good for step-like signals
//! - **Daubechies**: Orthogonal, maximum number of vanishing moments for given support
//! - **Symlets**: Nearly symmetric, modified Daubechies wavelets
//! - **Coiflets**: Nearly symmetric with vanishing moments for both scaling and wavelet functions
//!
//! ## Mathematical Properties
//!
//! Each wavelet is defined by a pair of filters:
//! - **Decomposition low-pass filter** h\[n\]: approximation coefficients
//! - **Decomposition high-pass filter** g\[n\]: detail coefficients
//! - **Reconstruction filters**: h'\[n\] and g'\[n\] for perfect reconstruction
//!
//! For orthogonal wavelets, reconstruction filters satisfy:
//! ```text
//! h'[n] = h[-n]
//! g'[n] = g[-n]
//! ```

use super::{WaveletError, WaveletResult};

/// Trait for wavelet implementations
///
/// All wavelets must provide decomposition and reconstruction filters
/// for the discrete wavelet transform.
pub trait Wavelet: Send + Sync {
    /// Get the decomposition low-pass filter (approximation)
    fn dec_lo(&self) -> &[f64];

    /// Get the decomposition high-pass filter (detail)
    fn dec_hi(&self) -> &[f64];

    /// Get the reconstruction low-pass filter
    fn rec_lo(&self) -> &[f64];

    /// Get the reconstruction high-pass filter
    fn rec_hi(&self) -> &[f64];

    /// Get the filter length
    fn filter_len(&self) -> usize {
        self.dec_lo().len()
    }

    /// Get the wavelet name
    fn name(&self) -> &str;

    /// Get the vanishing moments count
    fn vanishing_moments(&self) -> usize;

    /// Check if the wavelet is orthogonal
    fn is_orthogonal(&self) -> bool;

    /// Check if the wavelet is symmetric
    fn is_symmetric(&self) -> bool;
}

/// Enumeration of available wavelet types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletType {
    /// Haar wavelet (db1)
    Haar,
    /// Daubechies wavelets (db1-db10)
    Daubechies(u8),
    /// Symlet wavelets (sym2-sym8)
    Symlet(u8),
    /// Coiflet wavelets (coif1-coif5)
    Coiflet(u8),
}

impl WaveletType {
    /// Create a wavelet instance from the type
    pub fn create(&self) -> WaveletResult<Box<dyn Wavelet>> {
        match self {
            WaveletType::Haar => Ok(Box::new(HaarWavelet)),
            WaveletType::Daubechies(n) => {
                if *n < 1 || *n > 10 {
                    return Err(WaveletError::InvalidWavelet(format!(
                        "Daubechies order must be 1-10, got {}",
                        n
                    )));
                }
                Ok(Box::new(DaubechiesWavelet::new(*n)))
            }
            WaveletType::Symlet(n) => {
                if *n < 2 || *n > 8 {
                    return Err(WaveletError::InvalidWavelet(format!(
                        "Symlet order must be 2-8, got {}",
                        n
                    )));
                }
                Ok(Box::new(SymletWavelet::new(*n)))
            }
            WaveletType::Coiflet(n) => {
                if *n < 1 || *n > 5 {
                    return Err(WaveletError::InvalidWavelet(format!(
                        "Coiflet order must be 1-5, got {}",
                        n
                    )));
                }
                Ok(Box::new(CoifletWavelet::new(*n)))
            }
        }
    }

    /// Get the name of the wavelet
    pub fn name(&self) -> String {
        match self {
            WaveletType::Haar => "haar".to_string(),
            WaveletType::Daubechies(n) => format!("db{}", n),
            WaveletType::Symlet(n) => format!("sym{}", n),
            WaveletType::Coiflet(n) => format!("coif{}", n),
        }
    }
}

/// Haar wavelet implementation
///
/// The Haar wavelet is the simplest wavelet and is equivalent to db1.
/// It has one vanishing moment and is both orthogonal and symmetric.
pub struct HaarWavelet;

impl Wavelet for HaarWavelet {
    fn dec_lo(&self) -> &[f64] {
        const HAAR_DEC_LO: [f64; 2] = [
            0.7071067811865476, // 1/√2
            0.7071067811865476, // 1/√2
        ];
        &HAAR_DEC_LO
    }

    fn dec_hi(&self) -> &[f64] {
        const HAAR_DEC_HI: [f64; 2] = [
            -0.7071067811865476, // -1/√2
            0.7071067811865476,  // 1/√2
        ];
        &HAAR_DEC_HI
    }

    fn rec_lo(&self) -> &[f64] {
        self.dec_lo()
    }

    fn rec_hi(&self) -> &[f64] {
        const HAAR_REC_HI: [f64; 2] = [
            0.7071067811865476,  // 1/√2
            -0.7071067811865476, // -1/√2
        ];
        &HAAR_REC_HI
    }

    fn name(&self) -> &str {
        "haar"
    }

    fn vanishing_moments(&self) -> usize {
        1
    }

    fn is_orthogonal(&self) -> bool {
        true
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}

/// Daubechies wavelet implementation
///
/// Daubechies wavelets are orthogonal with compact support and maximum
/// number of vanishing moments for a given support width.
pub struct DaubechiesWavelet {
    order: u8,
    dec_lo: Vec<f64>,
    dec_hi: Vec<f64>,
    rec_lo: Vec<f64>,
    rec_hi: Vec<f64>,
}

impl DaubechiesWavelet {
    /// Create a new Daubechies wavelet of given order (1-10)
    pub fn new(order: u8) -> Self {
        let dec_lo = Self::get_dec_lo_filter(order);
        let len = dec_lo.len();

        // Compute high-pass filter using QMF relationship
        let mut dec_hi = vec![0.0; len];
        for (i, &val) in dec_lo.iter().enumerate() {
            dec_hi[len - 1 - i] = if i % 2 == 0 { val } else { -val };
        }

        // For orthogonal wavelets, reconstruction is time-reversed
        let mut rec_lo = dec_lo.clone();
        rec_lo.reverse();

        let mut rec_hi = dec_hi.clone();
        rec_hi.reverse();

        Self {
            order,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        }
    }

    /// Get decomposition low-pass filter coefficients
    fn get_dec_lo_filter(order: u8) -> Vec<f64> {
        match order {
            1 => vec![0.7071067811865476, 0.7071067811865476],
            2 => vec![
                -0.12940952255092145,
                0.22414386804185735,
                0.836516303737469,
                0.48296291314469025,
            ],
            3 => vec![
                0.035226291882100656,
                -0.08544127388224149,
                -0.13501102001039084,
                0.4598775021193313,
                0.8068915093133388,
                0.3326705529509569,
            ],
            4 => vec![
                -0.010597401784997278,
                0.032883011666982945,
                0.030841381835986965,
                -0.18703481171888114,
                -0.02798376941698385,
                0.6308807679295904,
                0.7148465705525415,
                0.23037781330885523,
            ],
            5 => vec![
                0.003335725285001549,
                -0.012580751999015526,
                -0.006241490213011705,
                0.07757149384006515,
                -0.03224486958502952,
                -0.24229488706619015,
                0.13842814590110342,
                0.7243085284385744,
                0.6038292697974729,
                0.160102397974125,
            ],
            6 => vec![
                -0.00107730108499558,
                0.004777257511010651,
                0.0005538422009938016,
                -0.031582039318031156,
                0.02752286553001629,
                0.09750160558707936,
                -0.12976686756709563,
                -0.22626469396516913,
                0.3152503517092432,
                0.7511339080215775,
                0.4946238903983854,
                0.11154074335008017,
            ],
            7 => vec![
                0.0003537138000010399,
                -0.0018016407039998328,
                0.00042957797300470274,
                0.012550998556013784,
                -0.01657454163101562,
                -0.03802993693503463,
                0.0806126091510659,
                0.07130921926705004,
                -0.22403618499416572,
                -0.14390600392910627,
                0.4697822874053586,
                0.7291320908465551,
                0.39653931948230575,
                0.07785205408506236,
            ],
            8 => vec![
                -0.00011747678400228192,
                0.0006754494059985568,
                -0.0003917403729959771,
                -0.00487035299301066,
                0.008746094047015655,
                0.013981027917015516,
                -0.04408825393106472,
                -0.01736930100202211,
                0.128747426620186,
                0.00047248457399797254,
                -0.2840155429624281,
                -0.015829105256023893,
                0.5853546836548691,
                0.6756307362980128,
                0.3128715909144659,
                0.05441584224308161,
            ],
            9 => vec![
                3.93473203162716025764e-05,
                -2.51963188942710123765e-04,
                2.30385763523195972796e-04,
                1.84764688305622654628e-03,
                -4.28150368246343025758e-03,
                -4.72320475775139716340e-03,
                2.23616621236790956428e-02,
                2.50947114831451972578e-04,
                -6.76328290613299742962e-02,
                3.07256814793333797586e-02,
                1.48540749338106375932e-01,
                -9.68407832229764564680e-02,
                -2.93273783279174915517e-01,
                1.33197385825007563742e-01,
                6.57288078051300517224e-01,
                6.04823123690111152939e-01,
                2.43834674612590340814e-01,
                3.80779473638783449996e-02,
            ],
            10 => vec![
                -1.32642028945212442831e-05,
                9.35886703200695919220e-05,
                -1.16466855129285448982e-04,
                -6.85856694959711618576e-04,
                1.99240529518505612994e-03,
                1.39535174705290106363e-03,
                -1.07331754833305745289e-02,
                3.60655356695616970131e-03,
                3.32126740593410019198e-02,
                -2.94575368218758133765e-02,
                -7.13941471663970816941e-02,
                9.30573646035723484049e-02,
                1.27369340335793251873e-01,
                -1.95946274377377049891e-01,
                -2.49846424327315380642e-01,
                2.81172343660577472857e-01,
                6.88459039453603538483e-01,
                5.27201188931725628350e-01,
                1.88176800077691497304e-01,
                2.66700579005555542256e-02,
            ],
            _ => vec![0.7071067811865476, 0.7071067811865476], // Default to db1
        }
    }
}

impl Wavelet for DaubechiesWavelet {
    fn dec_lo(&self) -> &[f64] {
        &self.dec_lo
    }

    fn dec_hi(&self) -> &[f64] {
        &self.dec_hi
    }

    fn rec_lo(&self) -> &[f64] {
        &self.rec_lo
    }

    fn rec_hi(&self) -> &[f64] {
        &self.rec_hi
    }

    fn name(&self) -> &str {
        "daubechies"
    }

    fn vanishing_moments(&self) -> usize {
        self.order as usize
    }

    fn is_orthogonal(&self) -> bool {
        true
    }

    fn is_symmetric(&self) -> bool {
        false
    }
}

/// Symlet wavelet implementation
///
/// Symlets are nearly symmetric modifications of Daubechies wavelets.
pub struct SymletWavelet {
    order: u8,
    dec_lo: Vec<f64>,
    dec_hi: Vec<f64>,
    rec_lo: Vec<f64>,
    rec_hi: Vec<f64>,
}

impl SymletWavelet {
    /// Create a new Symlet wavelet of given order (2-8)
    pub fn new(order: u8) -> Self {
        let dec_lo = Self::get_dec_lo_filter(order);
        let len = dec_lo.len();

        let mut dec_hi = vec![0.0; len];
        for (i, &val) in dec_lo.iter().enumerate() {
            dec_hi[len - 1 - i] = if i % 2 == 0 { val } else { -val };
        }

        let mut rec_lo = dec_lo.clone();
        rec_lo.reverse();

        let mut rec_hi = dec_hi.clone();
        rec_hi.reverse();

        Self {
            order,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        }
    }

    fn get_dec_lo_filter(order: u8) -> Vec<f64> {
        match order {
            2 => vec![
                -0.12940952255092145,
                0.22414386804185735,
                0.836516303737469,
                0.48296291314469025,
            ],
            3 => vec![
                0.035226291882100656,
                -0.08544127388224149,
                -0.13501102001039084,
                0.4598775021193313,
                0.8068915093133388,
                0.3326705529509569,
            ],
            4 => vec![
                -0.07576571478927333,
                -0.02963552764599851,
                0.49761866763201545,
                0.8037387518059161,
                0.29785779560527736,
                -0.09921954357684722,
                -0.012603967262037833,
                0.0322231006040427,
            ],
            5 => vec![
                0.027333068345077982,
                0.029519490925774643,
                -0.039134249302383094,
                0.1993975339773936,
                0.7234076904024206,
                0.6339789634582119,
                0.01660210576452232,
                -0.17532808990845047,
                -0.021101834024758855,
                0.019538882735286728,
            ],
            6 => vec![
                0.015404109327027373,
                0.0034907120842174702,
                -0.11799011114819057,
                -0.048311742585633,
                0.4910559419267466,
                0.787641141030194,
                0.3379294217276218,
                -0.07263752278646252,
                -0.021060292512300564,
                0.04472490177066578,
                0.0017677118642428036,
                -0.007800708325034148,
            ],
            7 => vec![
                2.68181456825787806544e-03,
                -1.04738488868291626348e-03,
                -1.26363034032519298833e-02,
                3.05155131659635703301e-02,
                6.78926935013726973178e-02,
                -4.95528349371272547330e-02,
                1.74412550868558273442e-02,
                5.36101917091762802947e-01,
                7.67764317003164054043e-01,
                2.88629631751514625915e-01,
                -1.40047240442961518081e-01,
                -1.07808237703817741404e-01,
                4.01024487153366342856e-03,
                1.02681767085112552601e-02,
            ],
            8 => vec![
                -3.38241595100612557276e-03,
                -5.42132331791148123872e-04,
                3.16950878114929807117e-02,
                7.60748732491760542435e-03,
                -1.43294238350809705063e-01,
                -6.12733590676585240797e-02,
                4.81359651258372212013e-01,
                7.77185751700523508312e-01,
                3.64441894835331403613e-01,
                -5.19458381077090372568e-02,
                -2.72190299170560028041e-02,
                4.91371796736075061585e-02,
                3.80875201389061510474e-03,
                -1.49522583370482308601e-02,
                -3.02920514721366799724e-04,
                1.88995033275946087807e-03,
            ],
            _ => vec![0.7071067811865476, 0.7071067811865476], // Default
        }
    }
}

impl Wavelet for SymletWavelet {
    fn dec_lo(&self) -> &[f64] {
        &self.dec_lo
    }

    fn dec_hi(&self) -> &[f64] {
        &self.dec_hi
    }

    fn rec_lo(&self) -> &[f64] {
        &self.rec_lo
    }

    fn rec_hi(&self) -> &[f64] {
        &self.rec_hi
    }

    fn name(&self) -> &str {
        "symlet"
    }

    fn vanishing_moments(&self) -> usize {
        self.order as usize
    }

    fn is_orthogonal(&self) -> bool {
        true
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}

/// Coiflet wavelet implementation
///
/// Coiflets have vanishing moments for both the scaling and wavelet functions.
pub struct CoifletWavelet {
    order: u8,
    dec_lo: Vec<f64>,
    dec_hi: Vec<f64>,
    rec_lo: Vec<f64>,
    rec_hi: Vec<f64>,
}

impl CoifletWavelet {
    /// Create a new Coiflet wavelet of given order (1-5)
    pub fn new(order: u8) -> Self {
        let dec_lo = Self::get_dec_lo_filter(order);
        let len = dec_lo.len();

        let mut dec_hi = vec![0.0; len];
        for (i, &val) in dec_lo.iter().enumerate() {
            dec_hi[len - 1 - i] = if i % 2 == 0 { val } else { -val };
        }

        let mut rec_lo = dec_lo.clone();
        rec_lo.reverse();

        let mut rec_hi = dec_hi.clone();
        rec_hi.reverse();

        Self {
            order,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        }
    }

    fn get_dec_lo_filter(order: u8) -> Vec<f64> {
        match order {
            1 => vec![
                -0.01565572813546454,
                -0.0727326195128539,
                0.38486484686420286,
                0.8525720202122554,
                0.3378976624578092,
                -0.07273261951284573,
            ],
            2 => vec![
                -7.20549445520346975788e-04,
                -1.82320887091103230049e-03,
                5.61143481936883428002e-03,
                2.36801719468477701869e-02,
                -5.94344186464310919593e-02,
                -7.64885990782807612121e-02,
                4.17005184423239083635e-01,
                8.12723635449413506215e-01,
                3.86110066822762887373e-01,
                -6.73725547237255945054e-02,
                -4.14649367868717769192e-02,
                1.63873364632036409849e-02,
            ],
            3 => vec![
                -3.45997731972727805847e-05,
                -7.09833025063790037977e-05,
                4.66216959820402881715e-04,
                1.11751877083063029875e-03,
                -2.57451768813679723533e-03,
                -9.00797613673062422257e-03,
                1.58805448636694518383e-02,
                3.45550275732977377197e-02,
                -8.23019271062998269972e-02,
                -7.17998216191548382925e-02,
                4.28483476377369998378e-01,
                7.93777222626087186619e-01,
                4.05176902409118244730e-01,
                -6.11233900029725524261e-02,
                -6.57719112814693640523e-02,
                2.34526961420771680455e-02,
                7.78259642567274631531e-03,
                -3.79351286438080192998e-03,
            ],
            4 => vec![
                -1.78499091449334685387e-06,
                -3.25964794003075104213e-06,
                3.12298615991952647304e-05,
                6.23388543127871916230e-05,
                -2.59974337122256815569e-04,
                -5.89020224633216536725e-04,
                1.26656107892566031047e-03,
                3.75143469714608662410e-03,
                -5.65828380013088348688e-03,
                -1.52117281876972109539e-02,
                2.50822533379496115380e-02,
                3.93344226055891491023e-02,
                -9.62204245359526422199e-02,
                -6.66274723668171670043e-02,
                4.34386033114356528984e-01,
                7.82238934424282605917e-01,
                4.15308427000682267582e-01,
                -5.60773196035692575445e-02,
                -8.12667102491937271003e-02,
                2.66823046696048303550e-02,
                1.60689471315750287417e-02,
                -7.34616793626805073686e-03,
                -1.62949242522678603741e-03,
                8.92313902537002971542e-04,
            ],
            5 => vec![
                -9.60401011276789414862e-08,
                -1.62379951720483375900e-07,
                2.06122039857887834848e-06,
                3.70072771133947961531e-06,
                -2.12702216725156142865e-05,
                -4.12198619242655009951e-05,
                1.40356328123732431454e-04,
                3.01857941668244784440e-04,
                -6.37558926125881154229e-04,
                -1.66162730392987881625e-03,
                2.43157544253828862210e-03,
                6.76152022062041693079e-03,
                -9.15950733867616252726e-03,
                -1.97583916009654650403e-02,
                3.26747994670573554954e-02,
                4.12875304721178337797e-02,
                -1.05563151307337232954e-01,
                -6.20377515749819599677e-02,
                4.37982306659163378448e-01,
                7.74293622860327435120e-01,
                4.21571266730754345975e-01,
                -5.20466702535547637298e-02,
                -9.19215880600860874017e-02,
                2.81697442705323534973e-02,
                2.34083221189277830565e-02,
                -1.01315848469002763726e-02,
                -4.15931262757864017576e-03,
                2.17829437784569472647e-03,
                3.58577741161757678184e-04,
                -2.12081862067494000086e-04,
            ],
            _ => vec![0.7071067811865476, 0.7071067811865476], // Default
        }
    }
}

impl Wavelet for CoifletWavelet {
    fn dec_lo(&self) -> &[f64] {
        &self.dec_lo
    }

    fn dec_hi(&self) -> &[f64] {
        &self.dec_hi
    }

    fn rec_lo(&self) -> &[f64] {
        &self.rec_lo
    }

    fn rec_hi(&self) -> &[f64] {
        &self.rec_hi
    }

    fn name(&self) -> &str {
        "coiflet"
    }

    fn vanishing_moments(&self) -> usize {
        2 * self.order as usize
    }

    fn is_orthogonal(&self) -> bool {
        true
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_wavelet() {
        let wavelet = HaarWavelet;
        assert_eq!(wavelet.filter_len(), 2);
        assert_eq!(wavelet.name(), "haar");
        assert_eq!(wavelet.vanishing_moments(), 1);
        assert!(wavelet.is_orthogonal());
        assert!(wavelet.is_symmetric());

        let dec_lo = wavelet.dec_lo();
        assert_eq!(dec_lo.len(), 2);

        // Check normalization
        let sum_sq: f64 = dec_lo.iter().map(|x| x * x).sum();
        assert!((sum_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_daubechies_creation() {
        for order in 1..=10 {
            let wavelet = DaubechiesWavelet::new(order);
            assert_eq!(wavelet.filter_len(), 2 * order as usize);
            assert!(wavelet.is_orthogonal());
            assert!(!wavelet.is_symmetric() || order == 1);

            // Check filter normalization
            let sum_sq: f64 = wavelet.dec_lo().iter().map(|x| x * x).sum();
            assert!((sum_sq - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symlet_creation() {
        for order in 2..=8 {
            let wavelet = SymletWavelet::new(order);
            assert!(wavelet.filter_len() >= 4);
            assert!(wavelet.is_orthogonal());
            assert!(wavelet.is_symmetric());

            let sum_sq: f64 = wavelet.dec_lo().iter().map(|x| x * x).sum();
            assert!((sum_sq - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_coiflet_creation() {
        for order in 1..=5 {
            let wavelet = CoifletWavelet::new(order);
            assert_eq!(wavelet.filter_len(), 6 * order as usize);
            assert!(wavelet.is_orthogonal());
            assert!(wavelet.is_symmetric());
            assert_eq!(wavelet.vanishing_moments(), 2 * order as usize);

            let sum_sq: f64 = wavelet.dec_lo().iter().map(|x| x * x).sum();
            assert!((sum_sq - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wavelet_type_creation() {
        let wavelet = WaveletType::Haar.create().expect("Failed to create Haar");
        assert_eq!(wavelet.name(), "haar");

        let wavelet = WaveletType::Daubechies(4)
            .create()
            .expect("Failed to create db4");
        assert_eq!(wavelet.filter_len(), 8);

        let wavelet = WaveletType::Symlet(3)
            .create()
            .expect("Failed to create sym3");
        assert!(wavelet.is_symmetric());

        let wavelet = WaveletType::Coiflet(2)
            .create()
            .expect("Failed to create coif2");
        assert_eq!(wavelet.vanishing_moments(), 4);
    }

    #[test]
    fn test_invalid_wavelet_orders() {
        assert!(WaveletType::Daubechies(0).create().is_err());
        assert!(WaveletType::Daubechies(11).create().is_err());
        assert!(WaveletType::Symlet(1).create().is_err());
        assert!(WaveletType::Symlet(9).create().is_err());
        assert!(WaveletType::Coiflet(0).create().is_err());
        assert!(WaveletType::Coiflet(6).create().is_err());
    }

    #[test]
    fn test_wavelet_type_name() {
        assert_eq!(WaveletType::Haar.name(), "haar");
        assert_eq!(WaveletType::Daubechies(3).name(), "db3");
        assert_eq!(WaveletType::Symlet(4).name(), "sym4");
        assert_eq!(WaveletType::Coiflet(2).name(), "coif2");
    }

    #[test]
    fn test_qmf_relationship() {
        // Test quadrature mirror filter relationship for db2
        let wavelet = DaubechiesWavelet::new(2);
        let dec_lo = wavelet.dec_lo();
        let dec_hi = wavelet.dec_hi();

        let len = dec_lo.len();
        for i in 0..len {
            let expected = if (len - 1 - i).is_multiple_of(2) {
                dec_lo[len - 1 - i]
            } else {
                -dec_lo[len - 1 - i]
            };
            assert!((dec_hi[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_orthogonality() {
        // Test orthogonality of Haar wavelet
        let wavelet = HaarWavelet;
        let dec_lo = wavelet.dec_lo();
        let dec_hi = wavelet.dec_hi();

        // Filters should be orthogonal
        let dot_product: f64 = dec_lo.iter().zip(dec_hi.iter()).map(|(a, b)| a * b).sum();
        assert!(dot_product.abs() < 1e-10);
    }
}
