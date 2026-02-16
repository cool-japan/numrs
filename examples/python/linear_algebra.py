#!/usr/bin/env python3
"""
NumRS2 Linear Algebra Example

Demonstrates linear algebra operations including matrix multiplication,
decompositions, and solving linear systems.
"""

import numrs2 as nr


def main():
    print("NumRS2 Linear Algebra Examples")
    print("=" * 60)
    print()

    # Matrix multiplication
    print("1. Matrix Multiplication")
    print("-" * 60)

    A = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
    B = nr.array([5.0, 6.0, 7.0, 8.0]).reshape([2, 2])

    print(f"Matrix A:\n{A.tolist()}")
    print(f"Matrix B:\n{B.tolist()}")

    C = nr.matmul(A, B)
    print(f"A @ B:\n{C.tolist()}")
    print()

    # Dot product
    print("2. Dot Product")
    print("-" * 60)

    u = nr.array([1.0, 2.0, 3.0])
    v = nr.array([4.0, 5.0, 6.0])

    dot_product = nr.dot(u, v)
    print(f"u: {u.tolist()}")
    print(f"v: {v.tolist()}")
    print(f"u · v: {dot_product}")
    print()

    # Determinant
    print("3. Determinant")
    print("-" * 60)

    M = nr.array([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
    print(f"Matrix M:\n{M.tolist()}")

    det = nr.linalg.det(M)
    print(f"det(M): {det}")
    print()

    # Matrix trace
    print("4. Matrix Trace")
    print("-" * 60)

    trace = nr.linalg.trace(M)
    print(f"trace(M): {trace}")
    print()

    # Matrix inversion
    print("5. Matrix Inversion")
    print("-" * 60)

    M_inv = nr.linalg.inv(M)
    print(f"M^(-1):\n{M_inv.tolist()}")

    # Verify M * M^(-1) ≈ I
    identity = nr.matmul(M, M_inv)
    print(f"M @ M^(-1) ≈ I:\n{identity.tolist()}")
    print()

    # Eigenvalues
    print("6. Eigenvalues")
    print("-" * 60)

    eigvals = nr.linalg.eigvals(M)
    print(f"Eigenvalues of M: {eigvals.tolist()}")
    print()

    # Eigendecomposition
    print("7. Eigendecomposition")
    print("-" * 60)

    vals, vecs = nr.linalg.eig(M)
    print(f"Eigenvalues: {vals.tolist()}")
    print(f"Eigenvectors shape: {vecs.shape}")
    print()

    # SVD
    print("8. Singular Value Decomposition (SVD)")
    print("-" * 60)

    A_rect = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
    print(f"Matrix A (2x3):\n{A_rect.tolist()}")

    U, S, Vt = nr.linalg.svd(A_rect)
    print(f"U shape: {U.shape}")
    print(f"S (singular values): {S.tolist()}")
    print(f"Vt shape: {Vt.shape}")
    print()

    # QR decomposition
    print("9. QR Decomposition")
    print("-" * 60)

    Q, R = nr.linalg.qr(A_rect)
    print(f"Q shape: {Q.shape}")
    print(f"R shape: {R.shape}")
    print()

    # Matrix norm
    print("10. Matrix Norm")
    print("-" * 60)

    vec = nr.array([3.0, 4.0])
    norm = nr.linalg.norm(vec)
    print(f"Vector: {vec.tolist()}")
    print(f"L2 norm: {norm}")
    print()

    # Condition number
    print("11. Condition Number")
    print("-" * 60)

    cond = nr.linalg.cond(M)
    print(f"Condition number of M: {cond}")
    print()

    print("=" * 60)
    print("Linear algebra examples completed successfully!")


if __name__ == "__main__":
    main()
