import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid function

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the Gaussian distributions
mu1 = np.array([-5, 5])
mu_neg1 = np.array([-2, 4])
Sigma = np.array([[2, 0], [0, 3]])


def generate_gaussian_data(n_samples):
    """Generate samples from two Gaussian distributions."""
    # Generate class 1 data
    X1 = np.random.multivariate_normal(mu1, Sigma, n_samples)
    y1 = np.ones(n_samples)

    # Generate class -1 data
    X_neg1 = np.random.multivariate_normal(mu_neg1, Sigma, n_samples)
    y_neg1 = -np.ones(n_samples)

    # Combine the data
    X = np.vstack((X1, X_neg1))
    y = np.concatenate((y1, y_neg1))

    return X, y


def logistic_regression_gd(X, y, learning_rate=0.01, n_iterations=1000):
    """Implement logistic regression using gradient descent."""
    n_samples, n_features = X.shape

    # Initialize weights and bias
    w = np.random.randn(n_features)
    b = np.random.randn()

    # Gradient descent
    for _ in range(n_iterations):
        # Forward pass
        z = np.dot(X, w) + b
        pred = expit(z)

        # Compute gradients
        dz = pred - (y + 1) / 2  # Convert y from {-1,1} to {0,1}
        dw = (1 / n_samples) * np.dot(X.T, dz)
        db = (1 / n_samples) * np.sum(dz)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


# Generate the dataset
n_samples = 5000
X, y = generate_gaussian_data(n_samples)

# Run multiple experiments and average the results
n_experiments = 100
w_accumulated = np.zeros(2)
b_accumulated = 0

for i in range(n_experiments):
    np.random.seed(i)  # Different seed for each experiment
    w, b = logistic_regression_gd(X, y)
    w_accumulated += w
    b_accumulated += b

# Compute averaged model parameters
w_avg = w_accumulated / n_experiments
b_avg = b_accumulated / n_experiments

# Compute optimal parameters (from part a)
Sigma_inv = np.linalg.inv(Sigma)
w_optimal = np.dot(Sigma_inv, (mu1 - mu_neg1))
b_optimal = -0.5 * (np.dot(np.dot(mu1.T, Sigma_inv), mu1) -
                    np.dot(np.dot(mu_neg1.T, Sigma_inv), mu_neg1))

# Plotting
plt.figure(figsize=(10, 8))

# Plot data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', alpha=0.5)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1', alpha=0.5)

# Plot decision boundaries
x1_range = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)

# Learned decision boundary
x2_learned = -(w_avg[0] * x1_range + b_avg) / w_avg[1]
plt.plot(x1_range, x2_learned, 'g-', label='Learned Boundary (Averaged)', linewidth=2)

# Optimal decision boundary
x2_optimal = -(w_optimal[0] * x1_range + b_optimal) / w_optimal[1]
plt.plot(x1_range, x2_optimal, 'k--', label='Optimal Boundary', linewidth=2)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression Decision Boundaries\n(Averaged over 100 experiments)')
plt.legend()
plt.grid(True)
plt.show()

# Print the parameters
print("Averaged Model Parameters:")
print(f"w = [{w_avg[0]:.4f}, {w_avg[1]:.4f}]")
print(f"b = {b_avg:.4f}")
print("\nOptimal Parameters:")
print(f"w = [{w_optimal[0]:.4f}, {w_optimal[1]:.4f}]")
print(f"b = {b_optimal:.4f}")