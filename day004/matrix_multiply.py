def matrix_multiply(M, N):
    """
    Custom matrix multiplication function using only basic Python
    M: I×J matrix (i rows by j columns) as a list of lists
    N: J×K matrix (j rows by k columns) as a list of lists
    Returns: P which is an I×K matrix as a list of lists
    """
    # Get dimensions
    I = len(M)  # Number of rows in M
    J = len(M[0])  # Number of columns in M (should equal rows in N)
    K = len(N[0])  # Number of columns in N

    # Verify that matrices can be multiplied
    if J != len(N):
        raise ValueError("Matrix dimensions don't match for multiplication")

    # Initialize result matrix P with zeros
    P = [[0 for _ in range(K)] for _ in range(I)]

    # Perform matrix multiplication
    for row in range(I):
        for col in range(K):
            # Calculate the inner product of row-th row of M and col-th column of N
            dot_product = 0
            for k in range(J):
                dot_product += M[row][k] * N[k][col]

            P[row][col] = dot_product

    return P


# Example usage
if __name__ == "__main__":
    # Example matrices
    M = [[1, 2, 3], [4, 5, 6]]  # 2×3 matrix (I=2, J=3)

    N = [[7, 8], [9, 10], [11, 12]]  # 3×2 matrix (J=3, K=2)

    # Perform matrix multiplication
    P = matrix_multiply(M, N)

    print("Matrix M (I×J):")
    for row in M:
        print(row)

    print("\nMatrix N (J×K):")
    for row in N:
        print(row)

    print("\nResult P (I×K):")
    for row in P:
        print(row)
