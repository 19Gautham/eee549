import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt

# FADE IN: Train function
def train(X, Y, lamb, epochs, lr=0.01):
    """
    X = (n * d)
    Y = (n * k), where k=10 i.e the number of classes
    lambda === lamb (I know, I just like to have a bit of fun with names)

    returns W_hat
    """

    # randomly initalize w & b
    # start of with a lambda of 0.1
    #no of dimensions
    # Using same notation as in the problem
    n, d = X.shape
    k = Y.shape[1]
    # k = 10

    # one column for each class, on row for each dimension
    w = np.random.randn(d, k)
    # it's a scalar, you have just one value for each class
    b = np.zeros((1, k))

    for i in range(epochs):
        regFormula = np.dot(X, w) + b
        # Softmax implementation, axis=1 means sum alongall columns for a row
        # keepdims=true to avoid the weird(60000,) issue
        sMax = np.exp(regFormula)/np.sum(np.exp(regFormula), axis=1, keepdims=True)

        dW = -np.dot(X.T, (Y - sMax)) / n + 2 * lamb * w
        db = -np.sum(Y - sMax, axis=0) / n

        # Update weights
        w -= lr * dW
        b -= lr * db

        if i%100 ==0:
            train_preds = predict(w, b, X)
            # print(f"Training predictions: {train_preds}")
            train_error = np.mean(train_preds != np.argmax(Y, axis=1))
            print(f"Epoch: {i+1}, Training Error: {train_error}")

    return w, b

def predict(W, b, X):
    """
    X: (m, d) test data
    W: trained weights
    b: trained bias
    Returns predicted labels for X.
    """
    Z = np.dot(X, W) + b
    predictions = np.argmax(Z, axis=1)
    return predictions


train_set = torchvision.datasets.FashionMNIST("./data", download=True)
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False)

X_train = train_set.data.numpy()
labels_train = train_set.targets.numpy()

X_test = test_set.data.numpy()
labels_test = test_set.targets.numpy()
sampleCount, width, height = X_train.shape
print(f"Sample Count: {sampleCount}, Height: {height}, Width: {width}")
plt.imshow(X_train[2])
plt.title(labels_train[0])
# plt.show()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
X_train = X_train/255.0
X_test = X_test/255.0

#given in question
lamb = 0.1
# just made this up tbh
iterations = 1000

# One hot encoding
Y_train = np.zeros((X_train.shape[0], 10))
Y_test = np.zeros((X_test.shape[0], 10))

labels_train = labels_train.reshape((labels_train.shape[0], 1))
labels_test = labels_test.reshape((labels_test.shape[0], 1))

for i in range(labels_train.shape[0]):
    Y_train[i][labels_train[i][0]] = 1

for i in range(labels_test.shape[0]):
    Y_test[i][labels_test[i][0]] = 1

W, b = train(X_train, Y_train, lamb, iterations)

# Test the model
train_predictions = predict(W, b, X_train)
test_predictions = predict(W, b, X_test)

# print(f"Training predictions shape: {train_preds.shape}")
# print(f"Testing prediction shape: {test_preds.shape}")

# argmax gives the index of the one hot encoded class, then we check it with the predicted class
train_error = np.mean(train_predictions != np.argmax(Y_train, axis=1))
test_error = np.mean(test_predictions != np.argmax(Y_test, axis=1))


print("\n\n")
print(f"Final Training Error: {train_error}")
print(f"Final Testing Error: {test_error}")