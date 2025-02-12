import numpy as np
from joblib import Parallel, delayed

"""
    The Coordinate Descent Algorithm (CD)
"""
def cd_algo(X, Y, lambda_val, stopping_condition, max_iterations, n_jobs=-1):
    # Number of samples
    n = X.shape[0]
    # Number of dimensions
    d = X.shape[1]

    W = np.zeros(d)
    b = 0
    a = 2 * np.sum(X ** 2, axis=0)

    # initializing residual, important to see that b=0 here so no point
    # using that here
    res_val = Y - np.dot(X, W)

    count = 0

    # yes, we do have a stopping condition but we need to have a limit on
    # the number of iterations that we run
    for _ in range(max_iterations):
        w_copy = np.copy(W)

        # Update each weight W[k] in parallel
        for k in range(d):

            r_plus_k = res_val + X[:, k] * W[k]
            ck = 2 * np.dot(X[:, k], r_plus_k)

            # Soft-thresholding
            if ck < -lambda_val:
                W[k] = (ck + lambda_val) / a[k]
            elif ck > lambda_val:
                W[k] = (ck - lambda_val) / a[k]
            else:
                W[k] = 0

            # getting back my residual value
            res_val = r_plus_k - X[:, k] * W[k]

        # Update the bias term b after the weight updates
        b = np.mean(Y - np.dot(X, W))

        count += 1

        if np.max(np.abs(W - w_copy)) < stopping_condition:
            print(f"Converged after {count} iterations for lambda = {lambda_val}")
            break

    return W, b


"""
    The Ridge Regression Algorithm (closed-form)
"""
def ridge_regression_algo(X, Y, lambda_val):
    d = X.shape[1]
    identity_matrix = np.eye(d)
    # just the standard (xT.x + LAMBDA.I)^-1 * (xT.y) formula
    return np.linalg.inv(np.dot(X.T, X) + (lambda_val * identity_matrix)) @ np.dot(X.T, Y)


def runner():
    n = 600
    d = 1200
    k = 120
    sigma = 1

    np.random.seed(42)
    # getting data from standard normal distribution
    X = np.random.normal(loc=0, scale=1, size=(n, d))
    w_true = np.concatenate([np.arange(1, k + 1) / k, np.zeros(d - k)])
    y = np.dot(X, w_true) + np.random.normal(0, sigma, n)

    # using the lambda max formula given in the question
    lambda_max = np.max(2 * np.abs(np.sum(X * (y - np.mean(y)).reshape(-1, 1), axis=0)))

    lambdas = [lambda_max / (1.5 ** i) for i in range(30)]

    # print(f"Lambda values: {lambdas}")

    nonzeros = []
    fd_list = []
    tp_list = []

    stopping_condition = 1e-6
    max_runs = 1000

    ridge_nonzeros = []
    ridge_norms = []

    # Run both Ridge and CD in the same loop
    for lambda_val in lambdas:
        ########## COORDINATE DESCENT PART #######################

        w_val, _ = cd_algo(X, y, lambda_val, stopping_condition, max_runs, n_jobs=-1)

        nonzero_count = np.sum(w_val != 0)
        nonzeros.append(nonzero_count)
        false_vals = np.sum((w_val != 0) & (w_true == 0)) / np.sum(w_val != 0) if np.sum(w_val != 0) > 0 else 0
        true_vals = np.sum((w_val != 0) & (w_true != 0)) / k
        fd_list.append(false_vals)
        tp_list.append(true_vals)

        ########## RIDGE REGRESSION PART #######################
        w_ridge = ridge_regression_algo(X, y, lambda_val)

        nonzero_count_ridge = np.sum(np.abs(w_ridge) > 1e-6)
        ridge_nonzeros.append(nonzero_count_ridge)
        l2_norm_ridge = np.linalg.norm(w_ridge, 2)
        ridge_norms.append(l2_norm_ridge)

    import matplotlib.pyplot as plt

    ### 4

    plt.figure()
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('# of non-zeros')
    plt.title('# of non-zero features vs Lambda (LASSO)')
    plt.show()

    print("\n\n")

    plt.figure()
    plt.plot(fd_list, tp_list, marker='o')
    plt.xlabel('False Discovery Rate (FDR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('FDR vs TPR (LASSO)')
    plt.show()

    print("\n\n")

    ## 5th part graphs

    plt.figure()
    plt.plot(lambdas, ridge_nonzeros)
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Number of non-zeros')
    plt.title('# of non-zero features vs Lambda (Ridge)')
    plt.show()

    print("\n\n")

    plt.figure()
    plt.plot(lambdas, ridge_norms, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('L2 Norm')
    plt.title('L2 Norm vs Lambda (Ridge)')
    plt.show()


if __name__ == "__main__":
    runner()
