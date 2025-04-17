use numrs2::prelude::*;

fn main() {
    println!("NumRS Array Comparison Operations Example");
    println!("========================================\n");

    // Create some arrays for demonstration
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::from_vec(vec![1.0000001, 2.0000002, 3.0000003, 5.0]);
    let c = Array::from_vec(vec![1.01, 2.02, 3.03, 4.04]);
    
    // 1. allclose - Testing approximate equality with default tolerances
    println!("1. Testing array approximate equality with allclose:");
    println!("A = {:?}", a.to_vec());
    println!("B = {:?}", b.to_vec());
    println!("C = {:?}", c.to_vec());
    println!("allclose(A, B) with default tolerances (rtol=1e-7, atol=0): {}", allclose(&a, &b));
    println!("allclose(A, C) with default tolerances: {}", allclose(&a, &c));
    
    // 2. allclose with custom tolerances
    println!("\n2. Testing with custom tolerances:");
    println!("allclose(A, C) with rtol=0.01, atol=0: {}", allclose_with_tol(&a, &c, 0.01, 0.0));
    
    // 3. isclose - Element-wise approximate equality
    println!("\n3. Testing element-wise approximate equality with isclose_array:");
    let result = isclose_array(&a, &c, 0.01, 0.0).unwrap();
    println!("isclose_array(A, C) with rtol=0.01, atol=0:");
    println!("{:?}", result.to_vec());
    
    // 4. array_equal - Exact equality test
    println!("\n4. Testing exact equality with array_equal:");
    let exact_a = Array::from_vec(vec![1, 2, 3, 4]);
    let exact_b = Array::from_vec(vec![1, 2, 3, 4]);
    let exact_c = Array::from_vec(vec![1, 2, 3, 5]);
    println!("array_equal for identical arrays: {}", array_equal(&exact_a, &exact_b));
    println!("array_equal for different arrays: {}", array_equal(&exact_a, &exact_c));
    
    // 5. Comparison operations (greater, less, etc.)
    println!("\n5. Comparison operations:");
    println!("A = {:?}", a.to_vec());
    println!("B = {:?}", b.to_vec());
    
    // greater
    let result = greater(&a, &b).unwrap();
    println!("greater(A, B): {:?}", result.to_vec());
    
    // less
    let result = less(&a, &b).unwrap();
    println!("less(A, B): {:?}", result.to_vec());
    
    // equal
    let result = equal(&a, &b).unwrap();
    println!("equal(A, B): {:?}", result.to_vec());
    
    // not_equal
    let result = not_equal(&a, &b).unwrap();
    println!("not_equal(A, B): {:?}", result.to_vec());
    
    // 6. Broadcasting with comparisons
    println!("\n6. Broadcasting with comparisons:");
    let row = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
    let col = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]);
    
    println!("row = {:?} with shape {:?}", row.to_vec(), row.shape());
    println!("col = {:?} with shape {:?}", col.to_vec(), col.shape());
    
    let result = greater_equal(&col, &row).unwrap();
    println!("greater_equal(col, row) (with broadcasting):");
    println!("Shape: {:?}", result.shape());
    println!("Values: {:?}", result.to_vec());
    
    // 7. all and any functions
    println!("\n7. all and any functions:");
    let all_true = Array::from_vec(vec![true, true, true]);
    let some_true = Array::from_vec(vec![false, true, false]);
    let none_true = Array::from_vec(vec![false, false, false]);
    
    println!("all([true, true, true]): {}", all(&all_true));
    println!("all([false, true, false]): {}", all(&some_true));
    println!("any([false, true, false]): {}", any(&some_true));
    println!("any([false, false, false]): {}", any(&none_true));
    
    // 8. Using boolean arrays for filtering
    println!("\n8. Using comparison results for filtering:");
    let data = Array::from_vec(vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0]);
    let threshold = Array::from_vec(vec![4.0]); // Using array for broadcasting
    
    println!("data = {:?}", data.to_vec());
    println!("Creating mask where data > 4.0");
    
    let mask = greater(&data, &threshold).unwrap();
    println!("mask = {:?}", mask.to_vec());
    
    // In NumPy, we'd do data[mask], but in NumRS we need to do this differently
    // We'll show the indices where the condition is true
    println!("Indices where data > 4.0:");
    for (i, &is_greater) in mask.to_vec().iter().enumerate() {
        if is_greater {
            println!("  Index {}: {}", i, data.get(&[i]).unwrap());
        }
    }
}