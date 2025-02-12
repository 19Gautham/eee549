import numpy as np
import math
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

    # randomly initialize w & b
    # start of with a lambda of 0.1
    # no of dimensions
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

# shoutout to the tansformation jutsu from Naruto here
# P.S if you read this, please ignore, this is how I have a bit of fun while coding:P
def transform(X_train, X_val, X_test, p_dim):
    # X = (n * d)
    d_dim = X_train.shape[1]
    G= np.random.normal(loc=0, scale=np.sqrt(0.005), size=(p_dim, d_dim))
    b = np.random.uniform(low=0, high=2*np.pi, size=(1, p_dim))
    transformed_X_train = np.sin(np.dot(X_train, G.T) + b)
    transformed_X_val = np.sin(np.dot(X_val, G.T) + b)
    transformed_X_test = np.sin(np.dot(X_test, G.T) + b)

    # (n * p) - output that we hope to get out there
    # so from (n * d) to (n * p)
    return transformed_X_train, transformed_X_val, transformed_X_test

def predict(W, b, X):

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

# plt.imshow(X_train[2])
# plt.title(labels_train[0])
# plt.show()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
X_train = X_train/255.0
X_test = X_test/255.0

# One hot encoding
Y_train = np.zeros((X_train.shape[0], 10))
Y_test = np.zeros((X_test.shape[0], 10))

labels_train = labels_train.reshape((labels_train.shape[0], 1))
labels_test = labels_test.reshape((labels_test.shape[0], 1))

for i in range(labels_train.shape[0]):
    Y_train[i][labels_train[i][0]] = 1

for i in range(labels_test.shape[0]):
    Y_test[i][labels_test[i][0]] = 1

#given in question
lamb = 0.1
# just made this up tbh
iterations = 1000
# projection dimension values
p_vals = [100, 500, 1000, 2000, 3000, 4000, 6000]
# p_vals = [p for p in range(100, 6000, 200)]

print(f"P vals: {p_vals}")
# exit(1)
# learning rate
lr = 0.01
# storage arrays
train_errors = []
val_errors = []
test_errors = []

from sklearn.model_selection import train_test_split

# https://scikit-learn.org/1.5/glossary.html#term-random_state
X_train, X_val, Y_train, Y_Val = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=42)

for p in p_vals:

    # first transform X values
    X_train_transformed, X_val_transformed, X_test_transformed = transform(X_train, X_val, X_test, p)
    # X_val_transformed = transform(X_val, p)
    # X_test_transformed = transform(X_test, p)

    # then we do the same old dance
    print(f"\n Projecting to {p} dimensions")
    W, b = train(X_train_transformed, Y_train, lamb, iterations)

    train_predictions = predict(W, b, X_train_transformed)
    train_error = np.mean(train_predictions != np.argmax(Y_train, axis=1))
    train_errors.append(train_error)

    val_predicitions = predict(W, b, X_val_transformed)
    val_error = np.mean(val_predicitions != np.argmax(Y_Val, axis=1))
    val_errors.append(val_error)

    test_predictions = predict(W, b, X_test_transformed)
    test_error = np.mean(test_predictions != np.argmax(Y_test, axis=1))
    test_errors.append(test_error)

    print(f"Training Error: {train_error}, Validation Error: {val_error}, Test Error: {test_error}\n")

# Plot training and validation errors
plt.figure(figsize=(8, 6))
plt.plot(p_vals, train_errors, label='Training Error', marker='o')
plt.plot(p_vals, val_errors, label='Validation Error', marker='o')
plt.plot(p_vals, test_errors, label='Test Error', marker='o')
plt.xlabel('p (dimension of transformed feature space)')
plt.ylabel('Error')
plt.title('Training, Validation, and Test Error vs. p')
plt.legend()
plt.grid(True)
plt.show()