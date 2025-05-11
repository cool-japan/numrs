use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS Stride Tricks Examples");
    println!("===========================\n");

    // SECTION 1: Basic Stride Manipulation
    println!("1. Basic Stride Manipulation");
    println!("---------------------------");

    // Create a 2D array
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    println!("Original array (shape {:?}):", array.shape());
    println!("{}", array);

    // Manipulate strides to select every other element in each dimension
    let strided = set_strides(&array, &[2, 2])?;
    println!(
        "\nStrided view with strides [2, 2] (shape {:?}):",
        strided.shape()
    );
    println!("{}", strided);

    // Create a reversed view (using a different approach since negative strides are tricky)
    let _reversed = array.clone(); // Just use a clone for the demo
    println!("\nUsing a manual approach for reversed arrays (could use indexing):");
    println!("Original array:");
    println!("{}", array);
    println!("We would access elements in reverse order along first axis");

    // SECTION 2: Advanced Strided Views with as_strided
    println!("\n2. Advanced Strided Views with as_strided");
    println!("---------------------------------------");

    // Create a 1D array
    let arr1d = Array::from_vec((0..8).collect::<Vec<i32>>());
    println!("Original 1D array:");
    println!("{}", arr1d);

    // Use as_strided to create a 2x4 array view of the data
    let view2d = as_strided(&arr1d, &[2, 4], &[4, 1])?;
    println!("\nAs 2x4 array (strides [4, 1]):");
    println!("{}", view2d);

    // Use as_strided to create a 4x2 array view of the data
    let view4x2 = as_strided(&arr1d, &[4, 2], &[2, 1])?;
    println!("\nAs 4x2 array (strides [2, 1]):");
    println!("{}", view4x2);

    // SECTION 3: Sliding Window Views
    println!("\n3. Sliding Window Views");
    println!("----------------------");

    // Create a 1D array for sliding windows
    let data = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    println!("Original 1D array:");
    println!("{}", data);

    // Create a sliding window view with window size 3 and step size 1
    let windows = sliding_window_view(&data, &[3], None)?;
    println!("\nSliding windows of size 3 (shape {:?}):", windows.shape());
    for i in 0..windows.shape()[0] {
        let window = windows.slice_view(0, i)?;
        println!("Window {}: {:?}", i, window.to_vec());
    }

    // Create a sliding window view with window size 3 and step size 2
    let windows_step2 = sliding_window_view(&data, &[3], Some(&[2]))?;
    println!(
        "\nSliding windows of size 3 with step 2 (shape {:?}):",
        windows_step2.shape()
    );
    for i in 0..windows_step2.shape()[0] {
        let window = windows_step2.slice_view(0, i)?;
        println!("Window {}: {:?}", i, window.to_vec());
    }

    // SECTION 4: Sliding Window Views on 2D Arrays
    println!("\n4. Sliding Window Views on 2D Arrays");
    println!("---------------------------------");

    // Create a 2D array
    let array2d = Array::from_vec((1..17).collect::<Vec<i32>>()).reshape(&[4, 4]);
    println!("Original 2D array:");
    println!("{}", array2d);

    // Create a 2x2 sliding window view
    let windows2d = sliding_window_view(&array2d, &[2, 2], None)?;
    println!("\n2x2 sliding windows (shape {:?}):", windows2d.shape());
    println!("This is a 4D array with shape [rows, cols, window_rows, window_cols]");

    // Print out each window
    for i in 0..windows2d.shape()[0] {
        for j in 0..windows2d.shape()[1] {
            let window = windows2d.index(&[
                IndexSpec::Index(i),
                IndexSpec::Index(j),
                IndexSpec::All,
                IndexSpec::All,
            ])?;
            println!("Window at position [{}, {}]:\n{}", i, j, window);
        }
    }

    // SECTION 5: Broadcasting with Stride Tricks
    println!("\n5. Broadcasting with Stride Tricks");
    println!("--------------------------------");

    // Create arrays of different shapes for broadcasting
    let a = Array::from_vec(vec![1, 2, 3]);
    let b = Array::from_vec(vec![10, 20, 30]).reshape(&[3, 1]);

    println!("Array a (shape {:?}):", a.shape());
    println!("{}", a);
    println!("\nArray b (shape {:?}):", b.shape());
    println!("{}", b);

    // Broadcast a to shape [3, 3]
    let a_broadcast = broadcast_to(&a, &[3, 3])?;
    println!("\nArray a broadcast to shape [3, 3]:");
    println!("{}", a_broadcast);

    // Broadcast b to shape [3, 3]
    let b_broadcast = broadcast_to(&b, &[3, 3])?;
    println!("\nArray b broadcast to shape [3, 3]:");
    println!("{}", b_broadcast);

    // Use the Array's built-in broadcasting functionality instead
    let a_broadcast = a.broadcast_to(&[3, 3])?;
    let b_broadcast = b.broadcast_to(&[3, 3])?;

    println!("\nBroadcast arrays result using built-in methods:");
    println!("Array 1 (shape {:?}):", a_broadcast.shape());
    println!("{}", a_broadcast);
    println!("Array 2 (shape {:?}):", b_broadcast.shape());
    println!("{}", b_broadcast);

    // Perform an operation using the broadcast arrays
    let sum = a_broadcast.add(&b_broadcast);
    println!("\nSum of broadcast arrays:");
    println!("{}", sum);

    // SECTION 6: Examining Byte Strides
    println!("\n6. Examining Byte Strides");
    println!("------------------------");

    // Create arrays with different layouts
    // For this example, just use the same array but with different logical layouts
    let c_array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    // Create a "column-major" like array by reshaping a clone
    let f_array = Array::from_vec(vec![1, 4, 2, 5, 3, 6]).reshape(&[2, 3]);

    println!("C-order array (shape {:?}):", c_array.shape());
    println!("{}", c_array);

    println!("\nColumn-major-like array (shape {:?}):", f_array.shape());
    println!("{}", f_array);

    // Get byte strides for both arrays
    let c_strides = byte_strides(&c_array);
    let f_strides = byte_strides(&f_array);

    println!("\nByte strides for C-order array: {:?}", c_strides);
    println!("Byte strides for column-major-like array: {:?}", f_strides);
    println!("The byte strides might be the same because we're using the same memory layout,");
    println!("but logically accessing the data in different orders.");

    Ok(())
}
