import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n = 256
sigma2 = 0.5
m_values = [1, 2, 4, 8, 16, 32]
x = np.arange(1, n + 1) / n


# Define the function g(x)
def g(x):
    return 2 * np.sin(10 * np.pi * x + 3) * np.exp((x - 1.5) ** 3)


# Generate yi with noise
# np.random.seed(0)  # For reproducibility
y = g(x) + np.random.normal(0, np.sqrt(sigma2), n)

# Store errors for each m
empirical_errors = []
bias_squared_errors = []
variance_errors = []
total_errors = []

# Step function estimator
for m in m_values:
    # Number of bins
    num_bins = n // m
    bin_width = m / n

    # Initialize arrays for bias and variance calculations
    bias_squared = 0
    variance = 0
    empirical_error = 0

    for j in range(1, num_bins + 1):
        # Indices of data points in the current bin
        bin_indices = np.arange((j - 1) * m, j * m)
        # Average of yi in the current bin
        c_j = np.mean(y[bin_indices])
        # Estimate g(x) in the current bin
        g_estimate = np.mean(g(x[bin_indices]))

        # Compute bias and variance
        bias_squared += np.sum((g_estimate - g(x[bin_indices])) ** 2)
        variance += np.sum((c_j - g_estimate) ** 2)

        # Empirical error
        empirical_error += np.sum((c_j - g(x[bin_indices])) ** 2)

    # Average bias-squared, variance, and empirical error
    bias_squared /= n
    variance /= n
    empirical_error /= n

    # Store results
    empirical_errors.append(empirical_error)
    bias_squared_errors.append(bias_squared)
    variance_errors.append(variance)
    total_errors.append(bias_squared + variance)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(m_values, empirical_errors, label='Average Empirical Error', marker='o')
plt.plot(m_values, bias_squared_errors, label='Average Bias-Squared', marker='o')
plt.plot(m_values, variance_errors, label='Average Variance', marker='o')
plt.plot(m_values, total_errors, label='Total Error (Bias + Variance)', marker='o')
plt.xlabel('m')
plt.ylabel('Error')
plt.title('Error Analysis for Different Values of m')
plt.legend()
plt.grid(True)
plt.show()
