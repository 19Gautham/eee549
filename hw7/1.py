import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import LeaveOneOut


# Generate the data
def generate_data(n=30):
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(0, 1, n)

    # True function f(x) = 4cos(4x)ln(x^2 + 1)
    def f(x): return 4 * np.cos(4 * x) * np.log(x ** 2 + 1)

    # Add noise from N(0,1)
    epsilon = np.random.normal(0, 1, n)
    y = f(x) + epsilon
    return x, y, f


# Define kernels
def polynomial_kernel(x1, x2, d):
    return (1 + np.dot(x1.reshape(-1, 1), x2.reshape(1, -1))) ** d


def rbf_kernel(x1, x2, gamma):
    diff = x1.reshape(-1, 1) - x2.reshape(1, -1)
    return np.exp(-gamma * (diff ** 2))


# Kernel Ridge Regression predictor
def krr_predict(x_train, x_test, y_train, kernel_func, kernel_param, lambda_reg):
    n = len(x_train)
    # Compute kernel matrix
    K = kernel_func(x_train, x_train, kernel_param)
    # Add regularization
    K_reg = K + lambda_reg * np.eye(n)
    # Solve for alpha
    alpha = np.linalg.solve(K_reg, y_train)
    # Compute test kernel matrix
    K_test = kernel_func(x_train, x_test, kernel_param)
    # Make predictions
    return np.dot(K_test.T, alpha)


# Leave-one-out cross validation
def loo_cv_error(x, y, kernel_func, kernel_param, lambda_reg):
    n = len(x)
    loo = LeaveOneOut()
    errors = []

    for train_idx, test_idx in loo.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pred = krr_predict(x_train, x_test, y_train, kernel_func, kernel_param, lambda_reg)
        errors.append((pred - y_test) ** 2)

    return np.mean(errors)


# Find optimal hyperparameters
def optimize_hyperparameters(x, y, kernel_func):
    def objective(params):
        kernel_param, lambda_reg = params
        return loo_cv_error(x, y, kernel_func, kernel_param, lambda_reg)

    # Different initial guesses for polynomial and RBF kernels
    if kernel_func == polynomial_kernel:
        bounds = [(1, 10), (1e-6, 1)]  # (d, lambda)
        init_guess = [3, 0.1]
    else:
        bounds = [(1e-3, 10), (1e-6, 1)]  # (gamma, lambda)
        init_guess = [1, 0.1]

    result = minimize(objective, init_guess, bounds=bounds, method='L-BFGS-B')
    return result.x


# Main execution
def main():
    # Generate data
    x, y, true_f = generate_data()

    # Find optimal hyperparameters
    poly_params = optimize_hyperparameters(x, y, polynomial_kernel)
    rbf_params = optimize_hyperparameters(x, y, rbf_kernel)

    print(f"Polynomial kernel - d: {poly_params[0]:.2f}, lambda: {poly_params[1]:.6f}")
    print(f"RBF kernel - gamma: {rbf_params[0]:.2f}, lambda: {rbf_params[1]:.6f}")

    # Generate points for plotting
    x_plot = np.linspace(0, 1, 200)
    y_true = true_f(x_plot)

    # Make predictions
    y_poly = krr_predict(x, x_plot, y, polynomial_kernel, poly_params[0], poly_params[1])
    y_rbf = krr_predict(x, x_plot, y, rbf_kernel, rbf_params[0], rbf_params[1])

    # Plot results
    plt.figure(figsize=(15, 6))

    # Polynomial kernel plot
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='red', label='Training data')
    plt.plot(x_plot, y_true, 'k--', label='True function')
    plt.plot(x_plot, y_poly, 'b-', label='Polynomial kernel')
    plt.title('Polynomial Kernel Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # RBF kernel plot
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='red', label='Training data')
    plt.plot(x_plot, y_true, 'k--', label='True function')
    plt.plot(x_plot, y_rbf, 'g-', label='RBF kernel')
    plt.title('RBF Kernel Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()