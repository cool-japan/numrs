use numrs2::prelude::*;
use numrs2::random;

fn main() -> Result<()> {
    println!("NumRS2 v0.1.0-alpha.4 Basic Usage Example");
    println!("==========================================");

    // Create arrays
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let b = Array::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[2, 3]);

    println!("Array a:");
    println!("{}", a);

    println!("\nArray b:");
    println!("{}", b);

    // Element-wise operations
    let c = a.add(&b);
    println!("\na + b = {}", c);

    let d = a.subtract(&b);
    println!("\na - b = {}", d);

    let e = a.multiply(&b);
    println!("\na * b = {}", e);

    let f = a.divide(&b);
    println!("\na / b = {}", f);

    // Matrix multiplication
    let g = a.matmul(&b.transpose())?;
    println!("\na @ b.T = {}", g);

    // Create arrays with built-in methods
    let zeros = Array::<f64>::zeros(&[2, 3]);
    println!("\nzeros(2, 3) = {}", zeros);

    let ones = Array::<f64>::ones(&[2, 3]);
    println!("\nones(2, 3) = {}", ones);

    let full = Array::<f64>::full(&[2, 3], 5.0);
    println!("\nfull(2, 3, 5.0) = {}", full);

    // Using array methods
    println!("\nArray information:");
    println!("shape: {:?}", a.shape());
    println!("ndim: {}", a.ndim());
    println!("size: {}", a.size());

    // Slicing
    let slice = a.slice(0, 1)?;
    println!("\nSlice a[1, :] = {}", slice);

    // Using map operations
    let squared = a.map(|x| x * x);
    println!("\nSquared elements: {}", squared);

    // Parallel operations
    let parallel_sqrt = a.par_map(|x: f64| x.sqrt());
    println!("\nParallel sqrt: {}", parallel_sqrt);

    // Random arrays
    let rand_arr = random::rand::<f64>(&[2, 3])?;
    println!("\nrandom(2, 3) = {}", rand_arr);

    // Vector operations
    let v1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Array::from_vec(vec![4.0, 5.0, 6.0]);
    let dot_product = v1.dot(&v2)?;
    println!("\nDot product: {}", dot_product);

    // Matrix condition number
    let well_conditioned = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
    let moderately_ill_conditioned =
        Array::from_vec(vec![1.0, 0.9999, 0.9999, 1.0]).reshape(&[2, 2]);
    let very_ill_conditioned = Array::from_vec(vec![1.0, 0.999999, 0.999999, 1.0]).reshape(&[2, 2]);

    println!("\nCondition Number Examples:");
    println!("Well-conditioned matrix:");
    println!("{}", well_conditioned);
    println!("Condition number: {}", well_conditioned.cond().unwrap());
    println!(
        "Is well-conditioned: {}",
        well_conditioned.is_well_conditioned().unwrap()
    );

    println!("\nModerately ill-conditioned matrix:");
    println!("{}", moderately_ill_conditioned);
    println!(
        "Condition number: {}",
        moderately_ill_conditioned.cond().unwrap()
    );
    println!(
        "Is well-conditioned: {}",
        moderately_ill_conditioned.is_well_conditioned().unwrap()
    );

    println!("\nVery ill-conditioned matrix:");
    println!("{}", very_ill_conditioned);
    println!("Condition number: {}", very_ill_conditioned.cond().unwrap());
    println!(
        "Is well-conditioned: {}",
        very_ill_conditioned.is_well_conditioned().unwrap()
    );

    Ok(())
}
