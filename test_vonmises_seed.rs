// Test script to check vonmises seed repeatability issue
use numrs2::random::distributions::{set_seed, normal};
use numrs2::random::advanced_distributions::vonmises;

fn main() {
    println!("Testing vonmises seed repeatability...");
    
    // Test with normal function first
    println!("\nTesting normal function:");
    set_seed(42);
    let normal1 = normal(0.0, 1.0, &[5]).unwrap();
    println!("First normal sequence: {:?}", normal1.to_vec());
    
    set_seed(42);
    let normal2 = normal(0.0, 1.0, &[5]).unwrap();
    println!("Second normal sequence: {:?}", normal2.to_vec());
    
    println!("Normal sequences identical: {}", normal1.to_vec() == normal2.to_vec());
    
    // Test with vonmises function
    println!("\nTesting vonmises function:");
    set_seed(42);
    let vonmises1 = vonmises(0.0, 1.0, &[5]).unwrap();
    println!("First vonmises sequence: {:?}", vonmises1.to_vec());
    
    set_seed(42);  
    let vonmises2 = vonmises(0.0, 1.0, &[5]).unwrap();
    println!("Second vonmises sequence: {:?}", vonmises2.to_vec());
    
    println!("Vonmises sequences identical: {}", vonmises1.to_vec() == vonmises2.to_vec());
    
    // Test the mixed case that might cause issues
    println!("\nTesting mixed case:");
    set_seed(42);
    let mixed_normal = normal(0.0, 1.0, &[3]).unwrap();
    let mixed_vonmises = vonmises(0.0, 1.0, &[3]).unwrap();
    println!("Mixed - normal: {:?}", mixed_normal.to_vec());
    println!("Mixed - vonmises: {:?}", mixed_vonmises.to_vec());
    
    set_seed(42);
    let mixed_normal2 = normal(0.0, 1.0, &[3]).unwrap();
    let mixed_vonmises2 = vonmises(0.0, 1.0, &[3]).unwrap();
    println!("Mixed2 - normal: {:?}", mixed_normal2.to_vec());
    println!("Mixed2 - vonmises: {:?}", mixed_vonmises2.to_vec());
    
    println!("Mixed normal sequences identical: {}", mixed_normal.to_vec() == mixed_normal2.to_vec());
    println!("Mixed vonmises sequences identical: {}", mixed_vonmises.to_vec() == mixed_vonmises2.to_vec());
}