/// Calculate the maximum absolute difference between two matrices
#[cfg(test)]
pub fn calculate_max_diff<T>(
    a: &crate::array::Array<T>,
    b: &crate::array::Array<T>,
) -> crate::error::Result<T>
where
    T: num_traits::Float + Clone + std::fmt::Debug,
{
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a != shape_b {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: shape_a,
            actual: shape_b,
        });
    }

    let mut max_diff = T::zero();

    for i in 0..shape_a[0] {
        for j in 0..shape_a[1] {
            let a_val = a.get(&[i, j])?;
            let b_val = b.get(&[i, j])?;
            let diff = num_traits::Float::abs(a_val - b_val);

            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    Ok(max_diff)
}
