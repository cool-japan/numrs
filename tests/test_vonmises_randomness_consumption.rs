// Test to verify vonmises random number consumption issue
use numrs2::random::advanced_distributions::vonmises;
use numrs2::random::distributions::{normal, set_seed};

#[test]
#[ignore = "Seeding behavior changed during SciRS2 migration - requires seeding implementation fix"]
fn test_vonmises_randomness_consumption() {
    println!("Testing vonmises random number consumption...");

    // Test scenario: Generate some vonmises values, then normal values
    // This should be reproducible if vonmises consumption is deterministic

    println!("\nScenario: vonmises followed by normal");

    set_seed(42);
    let vm1 = vonmises(0.0, 1.0, &[3]).unwrap();
    let norm1 = normal(0.0, 1.0, &[3]).unwrap();

    set_seed(42);
    let vm2 = vonmises(0.0, 1.0, &[3]).unwrap();
    let norm2 = normal(0.0, 1.0, &[3]).unwrap();

    println!("First run - vonmises: {:?}", vm1.to_vec());
    println!("First run - normal: {:?}", norm1.to_vec());
    println!("Second run - vonmises: {:?}", vm2.to_vec());
    println!("Second run - normal: {:?}", norm2.to_vec());

    assert_eq!(
        vm1.to_vec(),
        vm2.to_vec(),
        "Vonmises sequences should be identical"
    );
    assert_eq!(
        norm1.to_vec(),
        norm2.to_vec(),
        "Normal sequences should be identical"
    );

    // Test with different kappa values which might have different rejection rates
    println!("\nTesting with high kappa (more rejection sampling):");

    set_seed(123);
    let vm_high1 = vonmises(0.0, 10.0, &[2]).unwrap(); // High kappa = more concentrated
    let norm_after1 = normal(0.0, 1.0, &[2]).unwrap();

    set_seed(123);
    let vm_high2 = vonmises(0.0, 10.0, &[2]).unwrap();
    let norm_after2 = normal(0.0, 1.0, &[2]).unwrap();

    println!("High kappa first - vonmises: {:?}", vm_high1.to_vec());
    println!("High kappa first - normal: {:?}", norm_after1.to_vec());
    println!("High kappa second - vonmises: {:?}", vm_high2.to_vec());
    println!("High kappa second - normal: {:?}", norm_after2.to_vec());

    assert_eq!(
        vm_high1.to_vec(),
        vm_high2.to_vec(),
        "High kappa vonmises should be identical"
    );
    assert_eq!(
        norm_after1.to_vec(),
        norm_after2.to_vec(),
        "Normal after high kappa vonmises should be identical"
    );
}
