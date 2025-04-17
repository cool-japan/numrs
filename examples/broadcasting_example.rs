use numrs2::array::Array;
use numrs2::error::Result;

fn main() -> Result<()> {
    println!("NumRS2 Broadcasting Example");
    println!("=========================");
    
    // Create arrays with different shapes
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]);
    let b = Array::<f64>::from_vec(vec![4.0, 5.0, 6.0, 7.0]).reshape(&[1, 4]);
    
    println!("Array A (shape {:?}):", a.shape());
    println!("{}\n", a);
    
    println!("Array B (shape {:?}):", b.shape());
    println!("{}\n", b);
    
    // Broadcasting operations
    println!("Broadcasting Operations:");
    println!("----------------------");
    
    // Addition with broadcasting (result will be 3x4)
    let c = a.add_broadcast(&b)?;
    println!("A + B (shape {:?}):", c.shape());
    println!("{}\n", c);
    
    // Multiplication with broadcasting (result will be 3x4)
    let d = a.multiply_broadcast(&b)?;
    println!("A * B (shape {:?}):", d.shape());
    println!("{}\n", d);
    
    // More complex broadcasting example
    let e = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3]);
    let f = Array::<f64>::from_vec(vec![4.0, 5.0]).reshape(&[2, 1]);
    
    println!("Array E (shape {:?}):", e.shape());
    println!("{}\n", e);
    
    println!("Array F (shape {:?}):", f.shape());
    println!("{}\n", f);
    
    // Result should be shape (2, 3)
    let g = f.add_broadcast(&e)?;
    println!("F + E (shape {:?}):", g.shape());
    println!("{}\n", g);
    
    // Broadcasting with scalar operations
    println!("Scalar Operations:");
    println!("-----------------");
    
    let h = e.add_scalar(10.0);
    println!("E + 10.0:");
    println!("{}\n", h);
    
    let i = f.multiply_scalar(2.0);
    println!("F * 2.0:");
    println!("{}\n", i);
    
    Ok(())
}