use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Mul};

/// Represents a banded matrix, which is a sparse matrix with non-zero values
/// confined to a band around the main diagonal.
///
/// A banded matrix has all zero elements except for the main diagonal and a number
/// of diagonals above and below it.
#[derive(Clone)]
pub struct BandedMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Number of sub-diagonals (below main diagonal)
    sub_diagonals: usize,
    /// Number of super-diagonals (above main diagonal)
    super_diagonals: usize,
    /// Data storage for the band, in compact form
    data: Array<T>,
}

impl<T> BandedMatrix<T>
where
    T: Clone + Default + Zero + PartialEq,
{
    /// Create a new banded matrix
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `sub_diagonals` - Number of sub-diagonals (below main diagonal)
    /// * `super_diagonals` - Number of super-diagonals (above main diagonal)
    ///
    /// # Returns
    ///
    /// A new BandedMatrix instance with all elements set to default values
    pub fn new(rows: usize, cols: usize, sub_diagonals: usize, super_diagonals: usize) -> Self {
        // The data is stored in a compact form
        // We need (sub_diagonals + super_diagonals + 1) rows to store all the diagonals
        // Each row has length equal to the matrix width
        // Excess elements at the beginning/end of each row are not used (typically filled with zeros)
        let bands = sub_diagonals + super_diagonals + 1;
        let band_length = cols;

        // Create array with default values
        let data = Array::full(&[bands, band_length], T::default());

        Self {
            rows,
            cols,
            sub_diagonals,
            super_diagonals,
            data,
        }
    }

    /// Create a new banded matrix from an array
    ///
    /// # Arguments
    ///
    /// * `array` - A 2D array
    /// * `sub_diagonals` - Number of sub-diagonals (below main diagonal)
    /// * `super_diagonals` - Number of super-diagonals (above main diagonal)
    ///
    /// # Returns
    ///
    /// A new BandedMatrix instance with values copied from the array within the band
    ///
    /// # Errors
    ///
    /// Returns an error if the input array is not 2D
    pub fn from_array(
        array: &Array<T>,
        sub_diagonals: usize,
        super_diagonals: usize,
    ) -> Result<Self> {
        // Ensure array is 2D
        if array.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Banded matrix must be created from 2D array, got {}-dimensional array",
                array.ndim()
            )));
        }

        let shape = array.shape();
        let rows = shape[0];
        let cols = shape[1];

        // Create a new banded matrix with default values
        let mut banded = Self::new(rows, cols, sub_diagonals, super_diagonals);

        // Fill the banded matrix with values from the array. Bulk-acquire
        // `banded.data` once and inline `set()`'s band-index arithmetic
        // (duplicated from `BandedMatrix::set` below -- keep the two in sync
        // if that formula ever changes) so the whole O(rows * band_width)
        // pass pays `Arc::make_mut`'s unshare check once instead of once per
        // in-band element.
        let out = banded.data.array_mut();
        for i in 0..rows {
            for j in 0..cols {
                // Only consider elements within the band
                let diagonal = j as isize - i as isize;
                if diagonal >= -(sub_diagonals as isize) && diagonal <= super_diagonals as isize {
                    let value = array.get(&[i, j])?;
                    let band_row = (sub_diagonals as isize + diagonal) as usize;
                    let band_col = if diagonal < 0 { j } else { i };
                    out[[band_row, band_col]] = value;
                }
            }
        }

        Ok(banded)
    }

    /// Get the number of rows in the matrix
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns in the matrix
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Get the number of sub-diagonals (below main diagonal)
    pub fn sub_diagonals(&self) -> usize {
        self.sub_diagonals
    }

    /// Get the number of super-diagonals (above main diagonal)
    pub fn super_diagonals(&self) -> usize {
        self.super_diagonals
    }

    /// Get the band width (total width of the band)
    pub fn band_width(&self) -> usize {
        self.sub_diagonals + self.super_diagonals + 1
    }

    /// Check if the element at position (i, j) is within the band
    pub fn is_in_band(&self, i: usize, j: usize) -> bool {
        let diagonal = j as isize - i as isize;
        diagonal >= -(self.sub_diagonals as isize) && diagonal <= self.super_diagonals as isize
    }

    /// Get the value at position (i, j)
    ///
    /// # Arguments
    ///
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    ///
    /// The value at the specified position. If the position is outside the band,
    /// returns the default value for the type.
    ///
    /// # Errors
    ///
    /// Returns an error if indices are out of bounds
    pub fn get(&self, i: usize, j: usize) -> Result<T> {
        if i >= self.rows || j >= self.cols {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index ({}, {}) out of bounds for banded matrix with shape ({}, {})",
                i, j, self.rows, self.cols
            )));
        }

        // If outside the band, return default value
        if !self.is_in_band(i, j) {
            return Ok(T::default());
        }

        // Calculate the position in the compact storage format
        let diagonal = j as isize - i as isize;
        let band_row = (self.sub_diagonals as isize + diagonal) as usize;

        // For sub-diagonals, we need to offset the column index
        let band_col = if diagonal < 0 { j } else { i };

        Ok(self.data.get(&[band_row, band_col])?.clone())
    }

    /// Set the value at position (i, j)
    ///
    /// # Arguments
    ///
    /// * `i` - Row index
    /// * `j` - Column index
    /// * `value` - The value to set
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    ///
    /// # Errors
    ///
    /// Returns an error if indices are out of bounds or if trying to set a value outside the band
    pub fn set(&mut self, i: usize, j: usize, value: T) -> Result<()> {
        if i >= self.rows || j >= self.cols {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index ({}, {}) out of bounds for banded matrix with shape ({}, {})",
                i, j, self.rows, self.cols
            )));
        }

        // If outside the band, return error if value is not default
        if !self.is_in_band(i, j) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Cannot set element at ({}, {}), position is outside the band",
                i, j
            )));
        }

        // Calculate the position in the compact storage format
        let diagonal = j as isize - i as isize;
        let band_row = (self.sub_diagonals as isize + diagonal) as usize;

        // For sub-diagonals, we need to offset the column index
        let band_col = if diagonal < 0 { j } else { i };

        self.data.set(&[band_row, band_col], value)
    }

    /// Convert to a full (dense) Array
    pub fn to_array(&self) -> Array<T> {
        let mut array = Array::full(&[self.rows, self.cols], T::default());

        // `array` is write-only here (every read is `self.get`, a different
        // structure), so one bulk unshare covers the whole pass instead of
        // `Array::set`'s per-element `Arc::make_mut` check.
        let out = array.array_mut();
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.is_in_band(i, j) {
                    let value = self
                        .get(i, j)
                        .expect("to_array: index within band should be valid");
                    out[[i, j]] = value;
                }
            }
        }

        array
    }

    /// Get the diagonal elements as a vector
    pub fn diagonal(&self) -> Vec<T> {
        let diag_length = std::cmp::min(self.rows, self.cols);
        let mut diag = Vec::with_capacity(diag_length);

        for i in 0..diag_length {
            diag.push(
                self.get(i, i)
                    .expect("diagonal: index within min(rows, cols) should be valid"),
            );
        }

        diag
    }

    /// Check if the matrix is square
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

// Matrix multiplication for BandedMatrix
impl<T> BandedMatrix<T>
where
    T: Clone + Default + Zero + PartialEq + Add<Output = T> + Mul<Output = T>,
{
    /// Perform matrix-vector multiplication
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector to multiply with
    ///
    /// # Returns
    ///
    /// The result vector of the multiplication
    ///
    /// # Errors
    ///
    /// Returns an error if the vector length doesn't match the number of columns
    pub fn matvec(&self, vec: &[T]) -> Result<Vec<T>> {
        if vec.len() != self.cols {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![self.cols],
                actual: vec![vec.len()],
            });
        }

        let mut result = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            let mut sum = T::default();

            // Only need to sum over elements in the band
            let j_start = i.saturating_sub(self.sub_diagonals);
            let j_end = std::cmp::min(i + self.super_diagonals + 1, self.cols);

            for j in j_start..j_end {
                let a_ij = self
                    .get(i, j)
                    .expect("matvec: index within band should be valid");
                let x_j = vec[j].clone();
                sum = sum + (a_ij * x_j);
            }

            result.push(sum);
        }

        Ok(result)
    }
}

// Display implementation
impl<T> fmt::Display for BandedMatrix<T>
where
    T: Clone + fmt::Display + Default + Zero + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "BandedMatrix({}, {}, sub_diagonals={}, super_diagonals={})",
            self.rows, self.cols, self.sub_diagonals, self.super_diagonals
        )?;

        for i in 0..self.rows {
            write!(f, "[")?;

            for j in 0..self.cols {
                let value = self
                    .get(i, j)
                    .expect("Display: index within matrix bounds should be valid");

                if j > 0 {
                    write!(f, ", ")?;
                }

                // If it's outside the band (which means it's a default value),
                // display as '0' to make the pattern clearer
                if !self.is_in_band(i, j) {
                    write!(f, "0")?;
                } else {
                    write!(f, "{}", value)?;
                }
            }

            writeln!(f, "]")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `BandedMatrix<T>` had no test coverage anywhere in the crate despite
    // being public, even though `from_array()` and `to_array()` were
    // restructured for the W3-A2 COW-tax conversion -- added here to pin
    // down their round-trip behavior.

    #[test]
    fn test_from_array_round_trips_through_to_array() {
        // Tridiagonal 4x4: one sub-diagonal, one super-diagonal.
        let data = vec![
            1.0, 2.0, 0.0, 0.0, //
            3.0, 4.0, 5.0, 0.0, //
            0.0, 6.0, 7.0, 8.0, //
            0.0, 0.0, 9.0, 10.0,
        ];
        let a = Array::from_vec(data).reshape(&[4, 4]);

        let banded = BandedMatrix::from_array(&a, 1, 1).expect("tridiagonal input should fit");
        assert_eq!(banded.nrows(), 4);
        assert_eq!(banded.ncols(), 4);

        // Every element `from_array` copied in (i.e. every in-band position)
        // must read back exactly, both via `get()` and via `to_array()`.
        let round_tripped = banded.to_array();
        for i in 0..4 {
            for j in 0..4 {
                let expected = a.get(&[i, j]).expect("in-bounds get");
                assert_eq!(
                    banded.get(i, j).expect("in-bounds get"),
                    expected,
                    "BandedMatrix::get mismatch at ({i},{j})"
                );
                assert_eq!(
                    round_tripped.get(&[i, j]).expect("in-bounds get"),
                    expected,
                    "to_array mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_from_array_ignores_out_of_band_entries() {
        // A dense matrix with a nonzero far corner that a 0-bandwidth
        // (diagonal-only) banded matrix must drop.
        let a = Array::from_vec(vec![1.0, 99.0, 99.0, 2.0]).reshape(&[2, 2]);
        let banded = BandedMatrix::from_array(&a, 0, 0).expect("diagonal-only should fit");

        assert_eq!(banded.get(0, 0).expect("in-bounds get"), 1.0);
        assert_eq!(banded.get(1, 1).expect("in-bounds get"), 2.0);
        // Off-diagonal entries are outside the band, so they read back as the
        // type's default rather than the dropped input value.
        assert_eq!(banded.get(0, 1).expect("in-bounds get"), 0.0);
        assert_eq!(banded.get(1, 0).expect("in-bounds get"), 0.0);
    }
}
