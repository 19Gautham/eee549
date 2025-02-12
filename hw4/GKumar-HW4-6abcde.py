import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


# Coordinate Descent Algorithm (LASSO)
def cd_algo(X, Y, lambda_val, stopping_condition, max_iterations):
    n = X.shape[0]  # Number of samples
    d = X.shape[1]  # Number of dimensions

    W = np.zeros(d)
    a = 2 * np.sum(X ** 2, axis=0)

    residuals = Y - np.dot(X, W)
    count = 0

    for _ in range(max_iterations):
        w_copy = np.copy(W)

        # Update each weight W[k]
        for k in range(d):
            r_plus_k = residuals + X[:, k] * W[k]
            ck = 2 * np.dot(X[:, k], r_plus_k)

            if ck < -lambda_val:
                W[k] = (ck + lambda_val) / a[k]
            elif ck > lambda_val:
                W[k] = (ck - lambda_val) / a[k]
            else:
                W[k] = 0

            residuals = r_plus_k - X[:, k] * W[k]

        if np.max(np.abs(W - w_copy)) < stopping_condition:
            break
        count += 1

    return W

"""
    This is basically used to normalize the data
"""
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def runner():
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    lambda_max = np.max(2 * np.abs(np.dot(X_train_scaled.T, y_train - np.mean(y_train))))
    lambda_vals = [lambda_max / (1.5 ** i) for i in range(30)]

    # Initialize variables to store results
    nonzeros = []
    train_errors = []
    test_errors = []

    stopping_condition = 1e-6
    max_iterations = 1000

    for lambda_val in lambda_vals:
        # Solve with Coordinate Descent LASSO
        w_val = cd_algo(X_train_scaled, y_train, lambda_val, stopping_condition, max_iterations)

        nonzero_count = np.sum(w_val != 0)
        nonzeros.append(nonzero_count)

        # figuring out the train and test errors
        train_vals = np.dot(X_train_scaled, w_val)
        train_error = np.mean((y_train - train_vals) ** 2)

        test_vals = np.dot(X_test_scaled, w_val)
        test_error = np.mean((y_test - test_vals) ** 2)

        train_errors.append(train_error)
        test_errors.append(test_error)

    # 6a plot
    plt.figure()
    plt.plot(lambda_vals, nonzeros)
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('# of non-zero features')
    plt.title('# of non-zero features vs Lambda (LASSO)')
    plt.show()

    # 6b plot
    plt.figure()
    for i in range(X_train_scaled.shape[1]):
        plt.plot(lambda_vals,
                 [cd_algo(X_train_scaled, y_train, lambda_val, stopping_condition, max_iterations)[i] for lambda_val in
                  lambda_vals])
    plt.xscale('log')
    plt.xlabel('Lambda (on log scale)')
    plt.ylabel('Coefficient values')
    plt.title('Regularization Paths for LASSO Coefficients')
    plt.show()

    # 6c plot
    plt.figure()
    plt.plot(lambda_vals, train_errors, label='Train Error')
    plt.plot(lambda_vals, test_errors, label='Test Error')
    plt.xscale('log')
    plt.xlabel('Lambda (on log scale)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Train and Test Errors vs Lambda (LASSO)')
    plt.legend()
    plt.show()

    # 6d
    lambda_val = 4000
    w_lambda_val = cd_algo(X_train_scaled, y_train, lambda_val, stopping_condition, max_iterations)
    max_positive_coeff = np.argmax(w_lambda_val)
    max_negative_coeff = np.argmin(w_lambda_val)
    print(f"Largest positive coefficient index: {max_positive_coeff}, value: {w_lambda_val[max_positive_coeff]}")
    print(f"Largest negative coefficient index: {max_negative_coeff}, value: {w_lambda_val[max_negative_coeff]}")

    plt.figure()
    plt.bar(range(X_train_scaled.shape[1]), w_lambda_val)
    plt.title('Feature Coefficients (Lambda = 4000)')
    plt.xlabel('Feature Index Values')
    plt.ylabel('Coefficient Values')
    plt.show()

    # 6e
    k = 5
    # making use of the inbuilt K-FOLD CV function
    kf = KFold(n_splits=k)
    avg_val_errors = []
    std_errors = []

    for lambda_val in lambda_vals:
        k_fold_err = []
        for train_index, val_index in kf.split(X_train):
            # Split into train and validation sets
            X_k_fold_train, X_k_fold_val = X_train[train_index], X_train[val_index]
            y_k_fold_train, y_k_fold_val = y_train[train_index], y_train[val_index]

            X_k_fold_train_scaled, X_k_fold_val_scaled = scale_data(X_k_fold_train, X_k_fold_val)

            # Train the model using the coordinate descent LASSO
            w_k = cd_algo(X_k_fold_train_scaled, y_k_fold_train, lambda_val, stopping_condition, max_iterations)

            val_ = np.dot(X_k_fold_val_scaled, w_k)
            val_error = np.mean((y_k_fold_val - val_) ** 2)
            k_fold_err.append(val_error)

        # Calculate average validation error and standard error
        avg_val_error = np.mean(k_fold_err)
        avg_val_errors.append(avg_val_error)
        std_error = np.std(k_fold_err) / np.sqrt(k)
        std_errors.append(std_error)

    # Plot average validation error with error bars
    plt.figure()
    plt.errorbar(lambda_vals, avg_val_errors, yerr=std_errors, fmt='-o')
    plt.xscale('log')
    plt.xlabel('Lambda (on log scale)')
    plt.ylabel('Avg Val Error')
    plt.title('Avg Val Error vs Lambda')
    plt.show()

    # Getting the optimal lambda value by taking the lambda with the least corresponding val error
    optimal_lambda = lambda_vals[np.argmin(avg_val_errors)]
    print(f'Optimal lambda value: {optimal_lambda}')


if __name__ == "__main__":
    runner()