// Test for vonmises seed repeatability issue
use numrs2::random::advanced_distributions::vonmises;
use numrs2::random::distributions::{normal, set_seed};

#[test]
fn test_vonmises_seed_issue() {
    println!("Testing vonmises seed repeatability...");

    // Test with normal function first
    println!("\nTesting normal function:");
    set_seed(42);
    let normal1 = normal(0.0, 1.0, &[5]).unwrap();
    println!("First normal sequence: {:?}", normal1.to_vec());

    set_seed(42);
    let normal2 = normal(0.0, 1.0, &[5]).unwrap();
    println!("Second normal sequence: {:?}", normal2.to_vec());

    assert_eq!(
        normal1.to_vec(),
        normal2.to_vec(),
        "Normal sequences should be identical"
    );

    // Test with vonmises function
    println!("\nTesting vonmises function:");
    set_seed(42);
    let vonmises1 = vonmises(0.0, 1.0, &[5]).unwrap();
    println!("First vonmises sequence: {:?}", vonmises1.to_vec());

    set_seed(42);
    let vonmises2 = vonmises(0.0, 1.0, &[5]).unwrap();
    println!("Second vonmises sequence: {:?}", vonmises2.to_vec());

    assert_eq!(
        vonmises1.to_vec(),
        vonmises2.to_vec(),
        "Vonmises sequences should be identical"
    );
}
