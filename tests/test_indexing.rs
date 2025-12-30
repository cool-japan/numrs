use numrs2::array::Array;
use numrs2::indexing::*;

#[test]
fn test_ellipsis_indexing() {
    // Create a 3D array
    let mut array = Array::<f64>::zeros(&[2, 3, 4]);

    // Fill with test data
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let value = (i * 100 + j * 10 + k) as f64;
                array.set(&[i, j, k], value).unwrap();
            }
        }
    }

    // Test ellipsis at the beginning
    // This means: [:, :, 0] - select the first slice of the last dimension
    let result1 = array
        .index(&[IndexSpec::Ellipsis, IndexSpec::Index(0)])
        .unwrap();
    assert_eq!(result1.shape(), vec![2, 3]);

    // Check values
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(result1.get(&[i, j]).unwrap(), (i * 100 + j * 10) as f64);
        }
    }

    // Test ellipsis in the middle
    // This means: [0, :, :] - select the first slice of the first dimension
    let result2 = array
        .index(&[IndexSpec::Index(0), IndexSpec::Ellipsis])
        .unwrap();
    assert_eq!(result2.shape(), vec![3, 4]);

    // Check values
    for j in 0..3 {
        for k in 0..4 {
            assert_eq!(result2.get(&[j, k]).unwrap(), (j * 10 + k) as f64);
        }
    }

    // Test ellipsis with multiple indices
    // This means: [:, 0, 0] - select the first slice of second and third dimensions
    let result3 = array
        .index(&[
            IndexSpec::Ellipsis,
            IndexSpec::Index(0),
            IndexSpec::Index(0),
        ])
        .unwrap();
    assert_eq!(result3.shape(), vec![2]);

    // Check values
    for i in 0..2 {
        assert_eq!(result3.get(&[i]).unwrap(), (i * 100) as f64);
    }
}

#[test]
fn test_fancy_indexing() {
    // Create a 2D array
    let mut array = Array::<i32>::zeros(&[3, 4]);

    // Fill with test data
    for i in 0..3 {
        for j in 0..4 {
            let value = (i * 10 + j) as i32;
            array.set(&[i, j], value).unwrap();
        }
    }

    // Test fancy indexing with array of indices
    // Select rows 0 and 2
    let result1 = array
        .index(&[IndexSpec::Indices(vec![0, 2]), IndexSpec::All])
        .unwrap();
    assert_eq!(result1.shape(), vec![2, 4]);

    // Check values
    for j in 0..4 {
        assert_eq!(result1.get(&[0, j]).unwrap(), j as i32);
        assert_eq!(result1.get(&[1, j]).unwrap(), (20 + j) as i32);
    }

    // Test fancy indexing with array of indices for columns
    // Select columns 1 and 3
    let result2 = array
        .index(&[IndexSpec::All, IndexSpec::Indices(vec![1, 3])])
        .unwrap();
    assert_eq!(result2.shape(), vec![3, 2]);

    // Check values
    for i in 0..3 {
        assert_eq!(result2.get(&[i, 0]).unwrap(), (i * 10 + 1) as i32);
        assert_eq!(result2.get(&[i, 1]).unwrap(), (i * 10 + 3) as i32);
    }
}

#[test]
fn test_mask_indices() {
    // Test creating mask indices for upper triangle of a 3x3 matrix
    let indices = mask_indices(&[3, 3], |idx| idx[0] <= idx[1]).unwrap();

    // Check that we got indices for two dimensions
    assert_eq!(indices.len(), 2);

    // The first array contains row indices: [0,0,0,1,1,2]
    assert_eq!(indices[0].to_vec(), vec![0, 0, 0, 1, 1, 2]);

    // The second array contains column indices: [0,1,2,1,2,2]
    assert_eq!(indices[1].to_vec(), vec![0, 1, 2, 1, 2, 2]);
}

#[test]
fn test_ix_function() {
    // Create indices for a 2x3 grid
    let indices = indices_grid::<usize>(&[2, 3]).unwrap();

    // Check that we got indices for two dimensions
    assert_eq!(indices.len(), 2);

    // Check the shapes
    assert_eq!(indices[0].shape(), vec![2, 1]);
    assert_eq!(indices[1].shape(), vec![1, 3]);

    // Check values for first dimension: [[0], [1]]
    assert_eq!(indices[0].get(&[0, 0]).unwrap(), 0);
    assert_eq!(indices[0].get(&[1, 0]).unwrap(), 1);

    // Check values for second dimension: [[0, 1, 2]]
    assert_eq!(indices[1].get(&[0, 0]).unwrap(), 0);
    assert_eq!(indices[1].get(&[0, 1]).unwrap(), 1);
    assert_eq!(indices[1].get(&[0, 2]).unwrap(), 2);
}
