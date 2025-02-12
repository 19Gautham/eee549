import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Train function remains unchanged from previous code
def train(X, Y, lamb, epochs, lr=0.01):
    n, d = X.shape
    k = Y.shape[1]
    w = np.random.randn(d, k)
    b = np.zeros((1, k))

    for i in range(epochs):
        regFormula = np.dot(X, w) + b
        sMax = np.exp(regFormula) / np.sum(np.exp(regFormula), axis=1, keepdims=True)

        dW = -np.dot(X.T, (Y - sMax)) / n + 2 * lamb * w
        db = -np.sum(Y - sMax, axis=0) / n

        w -= lr * dW
        b -= lr * db

    return w, b

def predict(W, b, X):
    Z = np.dot(X, W) + b
    predictions = np.argmax(Z, axis=1)
    return predictions

# Random feature transformation
def random_feature_transform(X, p):
    d = X.shape[1]
    G = np.random.normal(0, np.sqrt(0.005), (p, d))
    b = np.random.uniform(0, 2 * np.pi, p)
    transformed_X = np.sin(np.dot(X, G.T) + b)
    return transformed_X

# Load FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST("./data", download=True)
X_train = train_set.data.numpy()
labels_train = train_set.targets.numpy()

# Reshape and normalize the data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2])) / 255.0

# One hot encoding for labels
Y_train = np.zeros((X_train.shape[0], 10))
for i in range(labels_train.shape[0]):
    Y_train[i][labels_train[i]] = 1

# Split data into training and validation set (80% train, 20% val)
X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Initialize parameters for cross-validation
lamb = 0.1
epochs = 1000
learning_rate = 0.01
p_values = [100, 500, 1000, 2000, 4000, 6000]  # Different values for p
train_errors = []
val_errors = []

# Cross-validation loop over p
for p in p_values:
    print(f"Training for p={p}...")

    # Apply feature transform
    X_train_transformed = random_feature_transform(X_train_split, p)
    X_val_transformed = random_feature_transform(X_val_split, p)

    # Train the model
    W, b = train(X_train_transformed, Y_train_split, lamb, epochs, lr=learning_rate)

    # Evaluate training error
    train_preds = predict(W, b, X_train_transformed)
    train_error = np.mean(train_preds != np.argmax(Y_train_split, axis=1))
    train_errors.append(train_error)

    # Evaluate validation error
    val_preds = predict(W, b, X_val_transformed)
    val_error = np.mean(val_preds != np.argmax(Y_val_split, axis=1))
    val_errors.append(val_error)

    print(f"p={p}, Training Error: {train_error}, Validation Error: {val_error}")

# Plot training and validation errors
plt.figure(figsize=(8, 6))
plt.plot(p_values, train_errors, label='Training Error', marker='o')
plt.plot(p_values, val_errors, label='Validation Error', marker='o')
plt.xlabel('p (dimension of transformed feature space)')
plt.ylabel('Error')
plt.title('Training and Validation Error vs. p')
plt.legend()
plt.grid(True)
plt.show()
