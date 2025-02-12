import numpy as np

def gen_full_rank_matrix(dim):
    matrix = np.random.randn(1, dim)
    rank = np.linalg.matrix_rank(matrix)
    # constructing a full rank matrix step by step, 1 row at a time
    while rank < dim:
        vector = np.random.randn(1, dim)
        temp = np.vstack([matrix, vector])
        if np.linalg.matrix_rank(temp) > rank:
            matrix = temp.copy()
            rank = np.linalg.matrix_rank(matrix)
    # print(f"Shape: {matrix.shape}")
    return matrix

# defining a dimension here - 3 by default
base_matrix = gen_full_rank_matrix(3)
covariance_matrix = np.dot(base_matrix.T, base_matrix)
print("Covariance matrix\n\n")
print(f"{covariance_matrix}\n\n")

# Check if matrix is symmetric or not
symmetric = True if np.allclose(covariance_matrix, covariance_matrix.T) else False

eigen_values = np.linalg.eigvals(covariance_matrix)
# semi-definite if all eigen values are greater than 0
positive_semi_definite = True if np.all(eigen_values > 0) else False

print(f"Covariance Matrix Transpose: \n\n{covariance_matrix.T}\n\n")
print(f"Eigen Values:\n\n {eigen_values}\n\n")

print(f"Symmetric: {symmetric}")
print(f"Positive semi-definite check: {positive_semi_definite}")