import numpy as np
import matplotlib.pyplot as plt

mu1 = np.array([-5, 5])
cov1 = np.array([[2, 0], [0, 3]])
mu2 = np.array([5, 5])

samples1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=5000)
samples2 = np.random.multivariate_normal(mean=mu2, cov=cov1, size=5000)

x_samples1 = samples1[:, 0]
y_samples1 = samples1[:, 1]

x_samples2 = samples2[:, 0]
y_samples2 = samples2[:, 1]

plt.figure(figsize=(10, 10))

plt.scatter(x_samples1, y_samples1, alpha=0.15, label='Class 1')
plt.scatter(x_samples2, y_samples2, alpha=0.15, label='Class -1')

w_optimal = np.array([-1.5, 1/3])
b_optimal = -81/12
x_values = np.linspace(-5, 5, 200)
# esentially trying to plot x2 by using the equation:
# -1.5x1 + 0.33x2 -6.75 = 0
y_values = -(w_optimal[0] * x_values + b_optimal) / w_optimal[1]

# plt.plot(x_values, y_values, color='red', label='Optimal Decision Boundary')


plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Optimal Decision Boundary for Gaussian Data')
plt.show()

from scipy.special import expit as sigmoid

def log_regression(X, y, lr=0.01, n_iter=10000):
    np.random.seed()
    w = np.random.randn(2)
    b = 0

    for _ in range(n_iter):
        f_x = np.dot(X, w) + b
        y_hat = sigmoid(f_x)
        errors = y - y_hat

        w += lr * np.dot(X.T, errors) / len(y)
        b += lr * np.sum(errors) / len(y)
    return w, b

X = np.vstack((samples1, samples2))
y = np.hstack((np.ones(5000), -np.ones(5000)))  # Labels: +1 for Class 1, -1 for Class 2

runs = 10
w_avg = np.zeros(2)
b_avg = 0
for _ in range(runs):
    w, b = log_regression(X, y)
    w_avg += w
    b_avg += b

w_avg = w_avg / runs
b_avg = b_avg / runs

y_values_lr = -((w_avg[0] * x_values) + b_avg) / w_avg[1]

plt.figure(figsize=(8, 6))
plt.scatter(x_samples1, y_samples1, alpha=0.15, label='Class 1')
plt.scatter(x_samples2, y_samples2, alpha=0.15, label='Class -1')
plt.plot(x_values, y_values, color='green', label='Optimal Separator')
plt.plot(x_values, y_values_lr, color='purple', linestyle='--', label='Averaged Logistic Regression Separator')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Logistic Regression Averaged Model vs Optimal Separator')
plt.show()
