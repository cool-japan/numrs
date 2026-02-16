#!/usr/bin/env python3
"""
NumRS2 Data I/O Example

Demonstrates saving and loading arrays in various formats.
"""

import numrs2 as nr
import tempfile
import os


def main():
    print("NumRS2 Data I/O Examples")
    print("=" * 60)
    print()

    # Create sample data
    data = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    matrix = data.reshape([2, 5])
    print(f"Sample data (2x5 matrix):")
    print(f"Shape: {matrix.shape}")
    print(f"Data: {matrix.tolist()}")
    print()

    # Create temporary directory for examples
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory: {tmpdir}")
        print()

        # NPY format
        print("1. NPY Format (NumPy Binary)")
        print("-" * 60)

        npy_file = os.path.join(tmpdir, "array.npy")
        print(f"Saving to {npy_file}...")
        nr.io.save_npy(npy_file, matrix)
        print(f"File size: {os.path.getsize(npy_file)} bytes")

        print("Loading from NPY...")
        loaded_npy = nr.io.load_npy(npy_file)
        print(f"Loaded shape: {loaded_npy.shape}")
        print(f"Loaded data: {loaded_npy.tolist()}")
        print()

        # CSV format
        print("2. CSV Format")
        print("-" * 60)

        csv_file = os.path.join(tmpdir, "array.csv")
        print(f"Saving to {csv_file}...")
        nr.io.save_csv(csv_file, data)  # Save flat array for CSV
        print(f"File size: {os.path.getsize(csv_file)} bytes")

        print("Loading from CSV...")
        loaded_csv = nr.io.load_csv(csv_file)
        print(f"Loaded shape: {loaded_csv.shape}")
        print(f"Loaded data: {loaded_csv.tolist()}")
        print()

        # JSON format
        print("3. JSON Format")
        print("-" * 60)

        json_file = os.path.join(tmpdir, "array.json")
        print(f"Saving to {json_file}...")
        nr.io.save_json(json_file, data)
        print(f"File size: {os.path.getsize(json_file)} bytes")

        print("Loading from JSON...")
        loaded_json = nr.io.load_json(json_file)
        print(f"Loaded shape: {loaded_json.shape}")
        print(f"Loaded data: {loaded_json.tolist()}")
        print()

        # Format comparison
        print("4. Format Comparison")
        print("-" * 60)

        formats = [
            ("NPY", npy_file),
            ("CSV", csv_file),
            ("JSON", json_file),
        ]

        for fmt_name, filepath in formats:
            size = os.path.getsize(filepath)
            print(f"{fmt_name:6s}: {size:6d} bytes")

        print()

    print("=" * 60)
    print("Data I/O examples completed successfully!")
    print()
    print("Note: All files were saved to a temporary directory")
    print("      and automatically cleaned up after the examples.")


if __name__ == "__main__":
    main()
