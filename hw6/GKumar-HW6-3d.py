import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

def log_regression(X, y, lr=0.01, iteration_count=2500):
    np.random.seed()
    w = np.random.randn(2)
    b = 0

    n_samples = len(y)

    for i in range(iteration_count):
        f_x = y * (np.dot(X, w) + b)
        a = sigmoid(f_x)

        error = 1 - a
        grad_w = -(1 / n_samples) * np.dot(X.T, y * error)
        grad_b = -(1 / n_samples) * np.sum(y * error)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

def calculate_stats(y, output, threshold):
    output = np.where(output >= threshold, 1, -1)

    tp = np.sum((y == 1) & (output == 1))
    fp = np.sum((y == -1) & (output == 1))
    tn = np.sum((y == -1) & (output == -1))
    fn = np.sum((y == 1) & (output == -1))

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


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

    output = sigmoid(np.dot(X, w_avg) + b)

    classification_thresholds = np.linspace(start=0, stop=1, num=50)
    tpr_list = []
    fpr_list = []

    for thresold in classification_thresholds:
        tpr, fpr = calculate_stats(y, output, thresold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, marker='o', color='b')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Curve (TPR vs FPR Curve for Varying Thresholds)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_prgm()