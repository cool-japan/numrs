#![allow(deprecated)]

// NumRS2 Type Conversion Example
//
// This example demonstrates the type conversion functionality in NumRS2,
// which allows converting arrays between different numeric types safely
// and efficiently. It shows various conversion methods, type promotions,
// and mixed-type operations.

use numrs2::prelude::*;
use scirs2_core::Complex;

fn main() -> Result<()> {
    println!("NumRS2 Type Conversion Example");
    println!("============================\n");

    // SECTION 1: Basic Type Conversions
    println!("1. Basic Type Conversions");
    println!("-----------------------");

    // Create arrays of different types
    let int_array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    let float_array = Array::from_vec(vec![1.5, 2.5, 3.5, 4.5]).reshape(&[2, 2]);

    println!("Integer array (i32):");
    println!("{}", int_array);

    println!("\nFloat array (f64):");
    println!("{}", float_array);

    // Convert between types using astype
    let int_to_float = int_array.astype::<f64>()?;
    println!("\nInteger array converted to f64:");
    println!("{}", int_to_float);
    println!("Notice that values maintain their original values but change type");

    let float_to_int = float_array.astype::<i32>()?;
    println!("\nFloat array converted to i32 (truncates decimal part):");
    println!("{}", float_to_int);
    println!("Notice that floating point values are truncated, not rounded");

    // Try a conversion that will fail
    let overflow_array = Array::from_vec(vec![1000, 2000, 3000]);
    println!("\nTrying to convert [1000, 2000, 3000] to i8 (range -128 to 127):");
    match overflow_array.astype::<i8>() {
        Ok(_) => println!("Conversion succeeded (unexpected)"),
        Err(e) => println!("Conversion failed as expected: {}", e),
    }
    println!("NumRS provides safe type conversions that fail when data would be lost");

    // SECTION 2: Safe Upcasting and Downcasting
    println!("\n2. Safe Upcasting and Downcasting");
    println!("-------------------------------");

    // Demonstrate upcasting (widening conversion)
    let small_ints = Array::from_vec(vec![100_i8, 120_i8, -100_i8, -120_i8]);
    println!("Small integers (i8): {:?}", small_ints.to_vec());

    let upcast_ints = small_ints.upcast::<i32>()?;
    println!("Upcast to i32: {:?}", upcast_ints.to_vec());
    println!("Upcasting is always safe as the target type can represent all source values");

    // Demonstrate downcasting (narrowing conversion)
    let safe_values = Array::from_vec(vec![100_i32, 50_i32, 25_i32, -100_i32]);
    println!("\nSafe integer values (i32): {:?}", safe_values.to_vec());

    let safe_downcast = safe_values.downcast::<i8>()?;
    println!("Safe downcast to i8: {:?}", safe_downcast.to_vec());
    println!("Downcast succeeds because all values fit in the target type");

    let unsafe_values = Array::from_vec(vec![100_i32, 200_i32, -100_i32, -200_i32]);
    println!(
        "\nUnsafe integer values (i32): {:?}",
        unsafe_values.to_vec()
    );

    match unsafe_values.downcast::<i8>() {
        Ok(_) => println!("Downcast succeeded (unexpected)"),
        Err(e) => println!("Downcast failed as expected: {}", e),
    }
    println!("Downcast fails because some values can't fit in i8 range (-128 to 127)");

    // Float to integer conversions
    let float_values = Array::from_vec(vec![1.1, 2.9, 3.5, 4.7]);
    println!("\nFloat values (f64): {:?}", float_values.to_vec());

    let float_to_int = float_values.astype::<i32>()?;
    println!("Converted to i32: {:?}", float_to_int.to_vec());
    println!("Note: Decimal parts are truncated, not rounded");

    // SECTION 3: Complex Number Conversions
    println!("\n3. Complex Number Conversions");
    println!("--------------------------");

    // Convert real numbers to complex
    let real_array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    println!("Real array (f64): {:?}", real_array.to_vec());

    let complex_array = real_array.to_complex::<f64>()?;
    println!("Converted to complex numbers: {:?}", complex_array.to_vec());
    println!("Each real value is converted to a complex number with zero imaginary part");

    // Create an array with mixed real and imaginary parts (for demonstration)
    let mut complex_nums = Vec::new();
    for val in real_array.to_vec() {
        complex_nums.push(Complex::new(val, val / 2.0));
    }
    let mixed_complex = Array::from_vec(complex_nums);
    println!(
        "\nComplex array with non-zero imaginary parts: {:?}",
        mixed_complex.to_vec()
    );

    // SECTION 4: Mixed-Type Operations
    println!("\n4. Mixed-Type Operations");
    println!("----------------------");

    // Create arrays of different types
    let int_array = Array::from_vec(vec![1, 2, 3]);
    let float_array = Array::from_vec(vec![1.5, 2.5, 3.5]);

    println!("Integer array (i32): {:?}", int_array.to_vec());
    println!("Float array (f64): {:?}", float_array.to_vec());

    // Perform mixed-type operations
    let addition = int_array.add_mixed::<f64, f64>(&float_array)?;
    println!(
        "\nMixed addition (int + float → f64): {:?}",
        addition.to_vec()
    );

    let subtraction = int_array.subtract_mixed::<f64, f64>(&float_array)?;
    println!(
        "Mixed subtraction (int - float → f64): {:?}",
        subtraction.to_vec()
    );

    let multiplication = int_array.multiply_mixed::<f64, f64>(&float_array)?;
    println!(
        "Mixed multiplication (int * float → f64): {:?}",
        multiplication.to_vec()
    );

    let division = int_array.divide_mixed::<f64, f64>(&float_array)?;
    println!(
        "Mixed division (int / float → f64): {:?}",
        division.to_vec()
    );

    println!("\nMixed-type operations automatically convert to an appropriate common type");
    println!("This behavior is similar to NumPy's type promotion rules");

    // SECTION 5: Type Promotion Rules
    println!("\n5. Type Promotion Rules");
    println!("---------------------");

    println!("NumRS uses a type precedence system similar to NumPy:");
    println!(" - bool (lowest)");
    println!(" - unsigned integers (u8, u16, u32, u64)");
    println!(" - signed integers (i8, i16, i32, i64)");
    println!(" - floating point (f32, f64)");
    println!(" - complex (Complex<f32>, Complex<f64>) (highest)");

    // Demonstrate type promotion with different combinations
    let bool_array = Array::from_vec(vec![true, false, true]);
    let u8_array = Array::from_vec(vec![1u8, 2u8, 3u8]);
    let i32_array = Array::from_vec(vec![10, 20, 30]);
    let f32_array = Array::from_vec(vec![1.0f32, 2.0f32, 3.0f32]);
    let f64_array = Array::from_vec(vec![1.5, 2.5, 3.5]);

    println!("\nConverting bool to u8 (can't use add_mixed directly on bool):");
    // bool型はNumCastを実装していないため、まずu8に変換する
    let bool_as_u8 = Array::from_vec(
        bool_array
            .to_vec()
            .iter()
            .map(|&b| if b { 1u8 } else { 0u8 })
            .collect::<Vec<u8>>(),
    );
    // 通常の加算演算子の代わりに要素ごとの加算を行う
    let mut bool_u8 = Array::zeros(&[bool_as_u8.size()]);
    for i in 0..bool_as_u8.size() {
        bool_u8.set(&[i], bool_as_u8.get(&[i])? + u8_array.get(&[i])?)?;
    }
    println!("Result: {:?}", bool_u8.to_vec());

    println!("\nPromoting u8 and i32 to i32:");
    let u8_i32 = u8_array.add_mixed::<i32, i32>(&i32_array)?;
    println!("Result: {:?}", u8_i32.to_vec());

    println!("\nPromoting i32 and f32 to f32:");
    let i32_f32 = i32_array.add_mixed::<f32, f32>(&f32_array)?;
    println!("Result: {:?}", i32_f32.to_vec());

    println!("\nPromoting f32 and f64 to f64:");
    let f32_f64 = f32_array.add_mixed::<f64, f64>(&f64_array)?;
    println!("Result: {:?}", f32_f64.to_vec());

    // SECTION 6: Converting Views
    println!("\n6. Type Conversion with Array Views");
    println!("--------------------------------");

    let original = Array::from_vec(vec![1, 2, 3, 4, 5]);
    let view = original.view();

    println!("Original array (i32): {:?}", original.to_vec());
    println!("View: {:?}", view.to_vec());

    // Convert view to a different type
    let converted_view = view.astype::<f64>()?;
    println!("\nView converted to f64: {:?}", converted_view.to_vec());

    // Convert view to complex
    let complex_view = view.to_complex::<f64>()?;
    println!("View converted to complex: {:?}", complex_view.to_vec());

    println!("\nWhen views are type-converted, the result is a new owned array");
    println!("This is necessary because the underlying data type has changed");

    // SECTION 7: Error Handling for Type Conversions
    println!("\n7. Error Handling for Type Conversions");
    println!("-----------------------------------");

    // Create an array with values that could overflow
    let large_values = Array::from_vec(vec![100, 200, 300]);

    // Try to convert to i8 (range: -128 to 127)
    println!("Trying to convert [100, 200, 300] to i8:");
    match large_values.astype::<i8>() {
        Ok(arr) => println!("Succeeded: {:?}", arr.to_vec()),
        Err(e) => println!("Failed: {}", e),
    }

    // Try to convert NaN or infinity to integers
    println!("\nTrying to convert NaN/infinity to integers:");
    let special_values = Array::from_vec(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    match special_values.astype::<i32>() {
        Ok(arr) => println!("Succeeded: {:?}", arr.to_vec()),
        Err(e) => println!("Failed: {}", e),
    }

    println!("\nNumRS provides error handling for unsafe conversions");
    println!("This prevents silent data corruption that could occur in other systems");

    Ok(())
}
