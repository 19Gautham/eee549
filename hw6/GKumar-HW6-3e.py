import numpy as np

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

    w = np.zeros(2)
    b = 0
    epochs = 1000

    for epoch in range(epochs):

        for i in range(len(y)):
            x = X[i]
            label = y[i]
            activation = np.dot(w, x) + b

            if label * activation <= 0:
                w += label * x
                b += label

    epoch = 0
    while True and epoch < epochs:

        flag = True

        for i in range(len(y)):
            x = X[i]
            label = y[i]
            activation = np.dot(w, x) + b

            if label * activation <= 0:
                w += label * x
                b += label
                # log that there is an update
                flag = False

        if epoch %1000 == 0:
            print(f"Epoch {epoch + 1} completed")

        # if there is no update then break, otherwise continue
        if flag:
            break

        epoch += 1

    print(f"Epochs run: {epoch + 1}")
    print(f'Final weights: {w}')
    print(f'Final bias: {b}')

if __name__ == "__main__":
    main_prgm()