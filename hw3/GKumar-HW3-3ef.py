import numpy as np
import matplotlib.pyplot as plt

# Generate a full rank matrix and ensure a covariance matrix
def gen_full_rank_matrix(dim):
    matrix = np.random.randn(1, dim)
    rank = np.linalg.matrix_rank(matrix)

    while rank < dim:
        vector = np.random.randn(1, dim)
        temp = np.vstack([matrix, vector])
        if np.linalg.matrix_rank(temp) > rank:
            matrix = temp.copy()
            rank = np.linalg.matrix_rank(matrix)
    # print(f"Shape: {matrix.shape}")
    return matrix

def gen_covariance_matrix(rng, d):
    # Generate a random base matrix and construct a covariance matrix
    base_matrix = gen_full_rank_matrix(d)
    covariance_matrix = np.dot(base_matrix.T, base_matrix)
    # Normalizing covariance matrix
    covariance_matrix /= np.trace(covariance_matrix) / d
    return covariance_matrix
    #
    # A = rng.standard_normal(size=(d, d))
    # covariance_matrix = np.dot(A, A.T)
    # return covariance_matrix / np.trace(covariance_matrix) * d

#### closed form solution bit
def ridge_regression(X, Y, lamda, d):
    w_hat = np.linalg.solve(np.dot(X.T, X) + (lamda *np.eye(d)), np.dot(X.T, Y))
    return w_hat

def normalized_error(X, w, y):
    return np.linalg.norm(np.dot(X, w) - y) / np.linalg.norm(y)

def main_func(lambda_vals, trials):
    train_n = 100
    test_n = 1000
    d = 100

    rng = np.random.default_rng()

    # keeping the weight vector constant across runs to understand the variations due to data and noise alone
    a_true = rng.normal(0, 1, size=(d, 1))

    train_errors = np.zeros((trials, len(lambda_vals)))
    test_errors = np.zeros((trials, len(lambda_vals)))

    # Normalizing covariance matrix
    covariance_matrix = gen_covariance_matrix(rng, d)
    # Generate mean vector with standard normal distribution
    mu = rng.standard_normal(size=d)

    # Generate random training and test data
    sigma_noise = rng.uniform(0.3, 0.7)

    for trial in range(trials):

        X_train = rng.multivariate_normal(mu, covariance_matrix, size=train_n)
        y_train = X_train.dot(a_true) + np.random.normal(0, sigma_noise, size=(train_n, 1))

        X_test = rng.multivariate_normal(mu, covariance_matrix, size=test_n)
        y_test = X_test.dot(a_true) + np.random.normal(0, sigma_noise, size=(test_n, 1))


        for i, lamda in enumerate(lambda_vals):
            w_hat = ridge_regression(X_train, y_train, lamda, d)
            train_errors[trial, i] = normalized_error(X_train, w_hat, y_train)
            test_errors[trial, i] = normalized_error(X_test, w_hat, y_test)

    # taking average across all trials for particular labda values
    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vals, avg_train_errors, label="Train Error", marker='o')
    plt.plot(lambda_vals, avg_test_errors, label="Test Error", marker='o')
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Average Error")
    plt.title("Train/Test Error for Ridge Regression")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    lambda_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    trials = 30
    main_func(lambda_vals, trials)