import numpy as np
import matplotlib.pyplot as plt

def gen_full_rank_matrix(dim):
    matrix = np.zeros((dim, dim))
    rank = np.linalg.matrix_rank(matrix)
    # constructing a full rank matrix step by step, 1 row at a time
    while rank < dim:
        vector = np.random.randint(low=-3, high=4, size=(1, 2), dtype=int)
        temp = np.vstack([matrix, vector])
        if np.linalg.matrix_rank(temp) > rank:
            matrix = temp.copy()
            rank = np.linalg.matrix_rank(matrix)

    return matrix

# defining a dimension here - 2 by default
base_matrix = gen_full_rank_matrix(2)
# covariance_matrix = np.dot(base_matrix.T, base_matrix)
# the line below is generated from the line of code above
covariance_matrix = np.array([[10, -9], [-9, 13]])

print("Covariance matrix\n\n")
print(f"{covariance_matrix}\n\n")

mu = np.random.randint(low=-3, high=4, size=(2, 1), dtype=int)
print(f"Mean Vector: \n{mu}")

eigen_values, eigenvectors = np.linalg.eig(covariance_matrix)
print(f"Eigen Values:\n\n {eigen_values}\n\n")
print(f"Eigen Vectors: {eigenvectors}")
mu = mu.reshape(-1)
samples = np.random.multivariate_normal(mean=mu, cov=covariance_matrix, size=1000)

x_samples = samples[:, 0]
y_samples = samples[:, 1]

plt.figure(figsize=(10, 10))

plt.scatter(x_samples, y_samples, alpha=0.3, label='Samples')
plt.scatter(mu[0], mu[1], marker='x', label='Mean')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")

color_arr = ['red', 'black']
for i in range(2):
    # 2 x standard_Deviation
    scaled_e_vector = eigenvectors[:, i] * np.sqrt(eigen_values[i]) * 2
    plt.arrow(mu[0], mu[1],
             scaled_e_vector[0], scaled_e_vector[1],
             head_width=0.1, head_length=0.1,
             fc=color_arr[i], ec=color_arr[i],
             label=f'Eigenvector {i+1}')

# Ellipse for the confidence interval around the mean
theta = np.linspace(0, 2 * np.pi, 120)
circle = np.array([np.cos(theta), np.sin(theta)])

transformation_matrix = np.dot(eigenvectors, np.diag(np.sqrt(eigen_values) * 2))
# print(transformation_matrix.shape)
ellipse = transformation_matrix @ circle + mu.reshape(2, 1)

plt.plot(ellipse[0], ellipse[1], 'b--', label="Confidence Ellipse (2 Standard Deviations)")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Visualization of the Gaussian Distribution")
plt.show()

scale_factor = 2.0

diff = samples - mu
mahalanobis_dist = np.sqrt(np.sum(diff @ np.linalg.inv(covariance_matrix) * diff, axis=1))
points_within = np.sum(mahalanobis_dist <= scale_factor)
percentage = (points_within / len(samples)) * 100
print(f"\nPercentage of points within the ellipse: {percentage:.1f}%")
