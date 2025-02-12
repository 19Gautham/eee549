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

def stochastic_gd(X_train, y_train, X_test, y_test, lam, step_size, max_iter):
    n = len(y_train)
    w = np.zeros(X_train.shape[1])
    b = 0
    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []

    for i in range(max_iter):
        for j in range(n):
            idx = np.random.randint(n)
            x_i = X_train[idx].reshape(1, -1)
            y_i = y_train[idx]

            dW, dB = calc_gradient(w, b, x_i, np.array([y_i]), lam)

            w -= step_size * dW
            b -= step_size * dB

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


    lamb = 0.1
    lr = 0.01
    max_iter = 200

    train_loss_list, test_loss_list, train_err_list, test_err_list = stochastic_gd(
        X_train, y_train, X_test, y_test, lamb, lr, max_iter)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.title("Loss vs Iteration/Epoch (Stochastic Gradient Descent)")
    plt.xlabel("Iteration/Epoch")
    plt.ylabel("Loss Values")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_err_list, label='Train Error')
    plt.plot(test_err_list, label='Test Error')
    plt.title("Misclassification Error vs Epcoh/Iterations (Stochastic Gradient Descent)")
    plt.xlabel("Iteration/Epoch")
    plt.ylabel("Misclassification Error Rate")
    plt.legend()

    plt.show()
