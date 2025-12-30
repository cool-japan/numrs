// Test to compare global vonmises vs direct advanced_distributions vonmises
use numrs2::random::advanced_distributions::vonmises as direct_vonmises;
use numrs2::random::distributions::{set_seed, vonmises as global_vonmises};

#[test]
#[ignore = "Seeding behavior changed during SciRS2 migration - requires seeding implementation fix"]
fn test_global_vs_direct_vonmises() {
    println!("Comparing global vonmises vs direct vonmises...");

    // Test with global vonmises function
    println!("\nTesting global vonmises function:");
    set_seed(42);
    let global1 = global_vonmises(0.0, 1.0, &[3]).unwrap();

    set_seed(42);
    let global2 = global_vonmises(0.0, 1.0, &[3]).unwrap();

    println!("Global first: {:?}", global1.to_vec());
    println!("Global second: {:?}", global2.to_vec());

    assert_eq!(
        global1.to_vec(),
        global2.to_vec(),
        "Global vonmises should be reproducible"
    );

    // Test with direct vonmises function
    println!("\nTesting direct vonmises function:");
    set_seed(42);
    let direct1 = direct_vonmises(0.0, 1.0, &[3]).unwrap();

    set_seed(42);
    let direct2 = direct_vonmises(0.0, 1.0, &[3]).unwrap();

    println!("Direct first: {:?}", direct1.to_vec());
    println!("Direct second: {:?}", direct2.to_vec());

    assert_eq!(
        direct1.to_vec(),
        direct2.to_vec(),
        "Direct vonmises should be reproducible"
    );

    // Compare the two implementations
    println!("\nComparing implementations:");
    println!("Global: {:?}", global1.to_vec());
    println!("Direct: {:?}", direct1.to_vec());

    // Note: These might be different since they use different algorithms
    // We just want to verify both are internally consistent
}
