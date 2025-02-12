import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

def log_regression(X, y, lr=0.001, iteration_count=2500):
    np.random.seed()
    w = np.random.randn(2)
    b = 0

    n_samples = len(y)

    for i in range(iteration_count):
        # Compute activation
        f_x = y * (np.dot(X, w) + b)
        a = sigmoid(f_x)

        # Compute gradients
        error = 1 - a  # Adjustment for y in {-1, 1}
        grad_w = -(1 / n_samples) * np.dot(X.T, y * error)
        grad_b = -(1 / n_samples) * np.sum(y * error)

        # Update weights
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

def main_prgm():
    np.random.seed(0)
    n_samples = 5000
    mu1 = np.array([-5, 5])
    mu2 = np.array([-2, 4])
    cov1 = np.array([[2, 0], [0, 3]])

    samples1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=n_samples)
    samples2 = np.random.multivariate_normal(mean=mu2, cov=cov1, size=n_samples)
    X = np.vstack((samples1, samples2))
    y = np.hstack((np.ones(n_samples), -np.ones(n_samples)))

    n_runs = 100
    w_avg = np.zeros(2)
    b_avg = 0
    for _ in range(n_runs):
        w, b = log_regression(X, y)
        w_avg += w
        b_avg += b

    w_avg = w_avg / n_runs
    b_avg = b_avg / n_runs

    # Plot decision boundaries
    x_values = np.linspace(-10, 5, 200)
    y_values = -(w_avg[0] * x_values + b_avg) / w_avg[1]

    plt.figure(figsize=(10, 10))
    plt.scatter(samples1[:, 0], samples1[:, 1], alpha=0.15, label='Class 1', color='blue')
    plt.scatter(samples2[:, 0], samples2[:, 1], alpha=0.15, label='Class -1', color='orange')
    plt.plot(x_values, y_values, color='purple', label='Logistic Regression Decision Boundary', linewidth=2)

    # Plot the optimal decision boundary
    w_optimal = np.array([-1.5, 1/3])
    b_optimal = -81/12
    y_optimal = -(w_optimal[0] * x_values + b_optimal) / w_optimal[1]
    plt.plot(x_values, y_optimal, color='red', label='Optimal Decision Boundary')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Logistic Regression Separator vs. Bayes Optimal Separator')
    plt.show()

if __name__ == "__main__":
    main_prgm()
