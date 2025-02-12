import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.model_selection import LeaveOneOut
from sklearn.kernel_ridge import KernelRidge

def main_prgm():

    def cross_val_kernel_ridge(x_vals, y_vals, kernel, hyperparams):
        looCv = LeaveOneOut()
        opt_hyperparam = None
        min_error = float("inf")

        for val in hyperparams:
            err = []

            for train_indices, test_indices in looCv.split(x_vals):
                x_train, x_test = x_vals[train_indices], x_vals[test_indices]
                y_train, y_test = y_vals[train_indices], y_vals[test_indices]

                if kernel == "polynomial":
                    model = KernelRidge(kernel="polynomial", degree=val[0], alpha=val[1])
                elif kernel == "rbf":
                    model = KernelRidge(kernel="rbf", gamma=val[0], alpha=val[1])

                model.fit(x_train, y_train)
                y = model.predict(x_test)
                err.append((y - y_test) ** 2)

            avg_error = np.mean(err)

            if avg_error < min_error:
                min_error = avg_error
                opt_hyperparam = val

        return opt_hyperparam, min_error

    count = 30
    x_vals = np.random.uniform(low=0, high=1, size=count).reshape(-1, 1)
    epsilon_vals = np.random.normal(loc=0, scale=1, size=count)
    f_x = lambda x: 4 * np.cos(4 * np.pi * x) * np.log(x ** 2 + 1)
    y_vals = f_x(x_vals).flatten() + epsilon_vals

    degree_vals = np.linspace(start=1, stop=51, num=50).astype(int)
    alpha_vals = np.logspace(-8, 1, 50)
    gamma_vals = np.logspace(-5, 1, 50)

    polynomial_kernels = [(d, alpha) for d in degree_vals for alpha in alpha_vals]
    opt_param_poly, best_poly_error = cross_val_kernel_ridge(x_vals, y_vals, "polynomial", polynomial_kernels)
    print(f"Best Polynomial Kernel Hyperparameters: degree={opt_param_poly[0]}, alpha={opt_param_poly[1]}")
    print(f"Best Polynomial Kernel Error: {best_poly_error}")

    rbf_kernels = [(gamma, alpha) for gamma in gamma_vals for alpha in alpha_vals]
    opt_param_rbf, best_rbf_error = cross_val_kernel_ridge(x_vals, y_vals, "rbf", rbf_kernels)
    print(f"Best RBF Kernel Hyperparameters: gamma={opt_param_rbf[0]}, alpha={opt_param_rbf[1]}")
    print(f"Best RBF Kernel Error: {best_rbf_error}")

    x_grid_vals = np.linspace(0, 1, 100).reshape(-1, 1)

    model_poly = KernelRidge(kernel="polynomial", degree=opt_param_poly[0], alpha=opt_param_poly[1])
    model_poly.fit(x_vals, y_vals)
    y_poly_pred = model_poly.predict(x_grid_vals)

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid_vals, f_x(x_grid_vals), label="True function $f(x)$", color="black", linestyle="--")
    plt.scatter(x_vals, y_vals, label="Data", color="blue")
    plt.plot(x_grid_vals, y_poly_pred, label="Polynomial Kernel Ridge", color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Kernel Ridge Regression with Best Polynomial Kernel")
    plt.show()

    model_rbf = KernelRidge(kernel="rbf", gamma=opt_param_rbf[0], alpha=opt_param_rbf[1])
    model_rbf.fit(x_vals, y_vals)
    y_rbf_pred = model_rbf.predict(x_grid_vals)

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid_vals, f_x(x_grid_vals), label="True function $f(x)$", color="black", linestyle="--")
    plt.scatter(x_vals, y_vals, label="Data", color="blue")
    plt.plot(x_grid_vals, y_rbf_pred, label="RBF Kernel Ridge", color="green")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Kernel Ridge Regression with Best RBF Kernel")
    plt.show()


if __name__ == "__main__":
    main_prgm()
