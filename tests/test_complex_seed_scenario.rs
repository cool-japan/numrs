// Test complex seed scenarios that might reveal issues
use numrs2::random::distributions::{set_seed, normal};
use numrs2::random::advanced_distributions::vonmises;

#[test]
fn test_complex_seed_scenarios() {
    println!("Testing complex seed scenarios...");
    
    // Scenario 1: Mixed function calls with same seed
    println!("\nScenario 1: Mixed functions, same seed");
    set_seed(42);
    let normal_a = normal(0.0, 1.0, &[3]).unwrap();
    let vonmises_a = vonmises(0.0, 1.0, &[3]).unwrap();
    
    set_seed(42);
    let normal_b = normal(0.0, 1.0, &[3]).unwrap();
    let vonmises_b = vonmises(0.0, 1.0, &[3]).unwrap();
    
    println!("Normal A: {:?}", normal_a.to_vec());
    println!("Normal B: {:?}", normal_b.to_vec());
    println!("Vonmises A: {:?}", vonmises_a.to_vec());
    println!("Vonmises B: {:?}", vonmises_b.to_vec());
    
    assert_eq!(normal_a.to_vec(), normal_b.to_vec(), "Normal sequences should be identical");
    assert_eq!(vonmises_a.to_vec(), vonmises_b.to_vec(), "Vonmises sequences should be identical");
    
    // Scenario 2: Reset seed multiple times
    println!("\nScenario 2: Multiple seed resets");
    set_seed(123);
    let v1 = vonmises(0.0, 1.0, &[2]).unwrap();
    set_seed(456);
    let v2 = vonmises(0.0, 1.0, &[2]).unwrap();
    set_seed(123);
    let v3 = vonmises(0.0, 1.0, &[2]).unwrap();
    
    println!("Vonmises 123: {:?}", v1.to_vec());
    println!("Vonmises 456: {:?}", v2.to_vec());
    println!("Vonmises 123 again: {:?}", v3.to_vec());
    
    assert_eq!(v1.to_vec(), v3.to_vec(), "Same seed should produce identical sequences");
    assert_ne!(v1.to_vec(), v2.to_vec(), "Different seeds should produce different sequences");
    
    // Scenario 3: vonmises with special parameters
    println!("\nScenario 3: Vonmises with special parameters");
    set_seed(42);
    let vm_uniform = vonmises(0.0, 0.0, &[3]).unwrap(); // Should be uniform-like
    set_seed(42);
    let vm_uniform2 = vonmises(0.0, 0.0, &[3]).unwrap();
    
    println!("Vonmises uniform 1: {:?}", vm_uniform.to_vec());
    println!("Vonmises uniform 2: {:?}", vm_uniform2.to_vec());
    
    assert_eq!(vm_uniform.to_vec(), vm_uniform2.to_vec(), "Even uniform vonmises should be reproducible");
}