import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    mnist = fetch_openml('mnist_784')
    X, y = mnist.data, mnist.target.astype(int)

    mask = (y == 6) | (y == 9)
    X = X[mask]
    y = y[mask]
    y = np.where(y == 6, 1, -1)  # Y = 1 for 6's and Y = -1 for 9's

    return X, y

def loss_fn(w, b, X, y, lam):
    n = X.shape[0]
    log_likelihood = np.sum(np.log(1 + np.exp(-y * (b + X @ w))))
    reg_term = lam * np.sum(w**2) / 2
    return log_likelihood / n + reg_term

def calc_gradient(w, b, X, y, lamb):
    n = X.shape[0]
    mu = 1 / (1 + np.exp(-y * (b + np.dot(X, w))))
    dB = -np.sum(y * (1 - mu)) / n
    dW = -(np.dot(X.T, (y * (1 - mu)))) / n + lamb * w
    return dW, dB

def mini_batch_gd(X_train, y_train, X_test, y_test, lam, step_size, max_iter, batch_size):
    n = X_train.shape[0]
    w = np.zeros(X_train.shape[1])
    b = 0
    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []

    for i in range(max_iter):
        indices = np.random.permutation(n)
        X_train_mixed = X_train[indices]
        y_train_mixed = y_train[indices]

        for j in range(0, n, batch_size):
            x_batch = X_train_mixed[j:j + batch_size]
            y_batch = y_train_mixed[j:j + batch_size]

            dW, dB = calc_gradient(w, b, x_batch, y_batch, lam)

            w -= step_size * dW
            b -= step_size * dB

        # Compute losses and errors after each epoch
        train_loss = loss_fn(w, b, X_train, y_train, lam)
        test_loss = loss_fn(w, b, X_test, y_test, lam)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_pred = np.sign(b + X_train @ w)
        test_pred = np.sign(b + X_test @ w)
        train_error = np.mean(train_pred != y_train)
        test_error = np.mean(test_pred != y_test)
        train_errors.append(train_error)
        test_errors.append(test_error)

    return train_losses, test_losses, train_errors, test_errors

if __name__ == "__main__":

    X, y = load_data()

    # https://scikit-learn.org/1.5/glossary.html#term-random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    batch_size = 100
    lr = 0.01
    iterations = 200

    lamb = 0.1

    train_loss_list, test_loss_list, train_err_list, test_err_list = mini_batch_gd(
        X_train, y_train, X_test, y_test, lamb, lr, iterations, batch_size)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.title("Loss vs Iteration/Epoch (Mini-Batch GD)")
    plt.xlabel("Iteration/Epoch")
    plt.ylabel("Loss Values")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_err_list, label='Train Error')
    plt.plot(test_err_list, label='Test Error')
    plt.title("Misclassification Error vs Epoch/Iterations (Mini-Batch GD)")
    plt.xlabel("Iteration/Epoch")
    plt.ylabel("Misclassification Error Rate")
    plt.legend()

    plt.show()
