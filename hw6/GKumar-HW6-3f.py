import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_samples = 5000
mu1 = np.array([-5, 5])
mu2 = np.array([-2, 4])
# modified covariance
cov1 = np.array([[0.2, 0], [0, 0.3]])

samples1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=n_samples)
samples2 = np.random.multivariate_normal(mean=mu2, cov=cov1, size=n_samples)
X = np.vstack((samples1, samples2))
y = np.hstack((np.ones(n_samples), -np.ones(n_samples)))

w = np.zeros(2)
b = 0

def plot_decision_boundary(w, b, X, y, title):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    y_vals = - (w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Perceptron Decision Boundary')

    w_optimal = np.array([-15, 10/3])
    b_optimal = -405/6
    x_values = np.linspace(-5, 5, 200)
    # esentially trying to plot x2 by using the equation:
    # -15x1 + 3.33x2 -405/6 = 0
    y_values = -(w_optimal[0] * x_values + b_optimal) / w_optimal[1]

    plt.plot(x_values, y_values, color='black', label='Optimal Decision Boundary')

    plt.legend()
    plt.grid(True)
    plt.xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
    plt.ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)

    plt.show()


epoch = 0
while True:

    flag = True

    for i in range(len(y)):
        x = X[i]
        label = y[i]
        activation = np.dot(w, x) + b

        if label * activation <= 0:
            w += label * x
            b += label
            flag = False

    if epoch %1000 == 0:
        print(f"Epoch {epoch + 1} completed")

    if flag:
        break

    epoch += 1

print(f"Dataset partitioned at end of epoch {epoch + 1}")

print(f'Final weights: {w}')
print(f'Final bias: {b}')

plot_decision_boundary(w, b, X, y, f"Dataset separated after {epoch+1} epochs")