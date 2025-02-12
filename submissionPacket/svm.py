import argparse
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import itertools

def analyze_best_model(best_model, best_hyperparams, train_images, train_labels, val_images, val_labels, test_images,
                       test_labels):
    print(f"Best Hyperparameters: {best_hyperparams}")

    kernel, tol, c_val_opt = best_hyperparams

    evaluate_model(best_model, test_images, test_labels)

    class_accuracies = class_accuracy(best_model, test_images, test_labels, classes=[5, 6])
    print("Per-Class Accuracy:", class_accuracies)

    plot_roc(best_model, test_images, test_labels)

    plot_accuracy_vs_samples(best_model, train_images, train_labels, test_images, test_labels)

    # Including more values > 1
    c_values = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    train_acc_list, val_acc_list, test_acc_list = [], [], []

    for c_val in c_values:
        try:

            model = SVC(kernel=kernel, C=c_val, tol=tol)
            model.fit(train_images, train_labels)

            train_acc = model.score(train_images, train_labels)
            val_acc = model.score(val_images, val_labels)
            test_acc = model.score(test_images, test_labels)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

        except Exception as e:
            print(f"Error during retraining for C={c_val}: {e}")

    print(f"Train Accuracies: {train_acc_list}")
    print(f"Validation Accuracies: {val_acc_list}")
    print(f"Test Accuracies: {test_acc_list}")

    plot_hyperparameter_tuning(c_values, train_acc_list, val_acc_list, test_acc_list)

def evaluate_model(model, test_images, test_labels):
    # Get predictions
    predictions = model.predict(test_images)
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def class_accuracy(model, test_images, test_labels, classes):
    predictions = model.predict(test_images)
    class_accuracies = {}
    for class_label in classes:
        indices = (test_labels == class_label)
        class_accuracies[class_label] = np.mean(predictions[indices] == test_labels[indices])
    return class_accuracies


def plot_roc(model, test_images, test_labels):
    probs = model.predict_proba(test_images)[:, 1]
    roc_auc = roc_auc_score(test_labels, probs)
    fpr, tpr, _ = roc_curve(test_labels, probs, pos_label=6)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_hyperparameter_tuning(c_values, train_acc_list, val_acc_list, test_acc_list):
    plt.figure()
    plt.plot(c_values, train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(c_values, val_acc_list, label='Validation Accuracy', marker='o')
    plt.plot(c_values, test_acc_list, label='Test Accuracy', marker='o')
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. C')
    plt.xscale('log')
    plt.legend()
    plt.show()


def plot_accuracy_vs_samples(model, train_images, train_labels, test_images, test_labels):
    sample_sizes = np.linspace(100, len(train_images), 10, dtype=int)
    train_accuracies = []
    test_accuracies = []

    for size in sample_sizes:
        model.fit(train_images[:size], train_labels[:size])
        train_accuracies.append(model.score(train_images[:size], train_labels[:size]))
        test_accuracies.append(model.score(test_images, test_labels))

    plt.figure()
    plt.plot(sample_sizes, train_accuracies, label='Train Accuracy')
    plt.plot(sample_sizes, test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Train/Test Accuracy vs. Number of Samples')
    plt.legend()
    plt.show()

def get_data(path, required_labels, train=True):
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to PyTorch tensors
    ])

    dataset = datasets.MNIST(root=path, train=train, transform=transform, download=True)

    images, labels = [], []
    for imgs, label in dataset:
      if label in required_labels.keys():
          flattened = imgs.view(imgs.size(0), -1).numpy()  # Flatten images
          images.append(flattened)
          labels.append(label)

    images = np.vstack(images)
    labels = np.array(labels)
    return images, labels


def normalize_data(data, mean=None, std_dev=None):
    if mean is None or std_dev is None:
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)

    # to account for division by zero
    std_dev[std_dev == 0] = 1e-8

    normalized_data = (data - mean) / std_dev
    return mean, std_dev, normalized_data

def train_model(train_images, train_labels, kernel, tol, C):
    model = SVC(kernel=kernel, tol=tol, C=C)
    model.fit(train_images, train_labels)
    return model

def get_model_accuracy(model, test_images, test_labels):
    predictions = model.predict(test_images)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

def tune_model(train_images, train_labels, val_images, val_labels):
    bestAccuracy = 0
    bestKernel = None
    bestTol = None
    bestC = None
    bestModel = None
    best_hyperparams = None

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    tol_values = [1e-3, 0.5*1e-3, 1e-4]
    c_values = [1e-4, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.1, 1, 10, 100]

    for kernel, tol, c_val in itertools.product(kernels, tol_values, c_values):

        print(f"Hyperparameters: kernel={kernel}, tol={tol}, c_val={c_val}")

        try:
            model = train_model(train_images, train_labels, kernel, tol, c_val)
            val_accuracy = get_model_accuracy(model, val_images, val_labels)
            print(f"Validation Accuracy: {val_accuracy * 100}%")

            if val_accuracy > bestAccuracy:
                bestAccuracy = val_accuracy
                bestModel = model
                best_hyperparams = (kernel, tol, c_val)

        except Exception as e:
            print(f"Error with hyperparams: kernel={kernel}, tol={tol}, c_val={c_val} during training: {e}")

    return bestModel, best_hyperparams, bestAccuracy



def run_best_model():

    print("Running the model with the best hyperparameters...")

    # let's build it out from scratch

    # Define paths to data
    train_data_path = "./mnist_data/train"
    test_data_path = "./mnist_data/test"

    mnist_labels = {5: "5 - five", 6: "6 - six"}

    # first, get the data
    train_images, train_labels = get_data(train_data_path, mnist_labels, train=True)
    test_images, test_labels = get_data(test_data_path, mnist_labels, train=False)

    # split the data for training and validation
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

    # Normalize data
    mean, std_dev, normalized_train_data = normalize_data(train_images)
    _, _, normalized_test_data = normalize_data(test_images, mean, std_dev)
    _, _, normalized_val_data = normalize_data(val_images, mean, std_dev)

    kernel, tol, c_val = ('poly', 0.001, 10)

    model = SVC(kernel=kernel, tol=tol, C=c_val, probability=True)
    model.fit(normalized_train_data, train_labels)

    analyze_best_model(
        model,
        (kernel, tol, c_val),
        normalized_train_data, train_labels,
        normalized_val_data, val_labels,
        normalized_test_data, test_labels
    )


def train_and_evaluate():
    print("Training the model and generating performance statistics...")

    # let's build it out from scratch

    # Define paths to data
    train_data_path = "./mnist_data/train"
    test_data_path = "./mnist_data/test"

    mnist_labels = {5: "5 - five", 6: "6 - six"}

    # first, get the data
    train_images, train_labels = get_data(train_data_path, mnist_labels, train=True)
    test_images, test_labels = get_data(test_data_path, mnist_labels, train=False)

    # split the data for training and validation
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

    # Normalize data
    mean, std_dev, normalized_train_data = normalize_data(train_images)
    _, _, normalized_test_data = normalize_data(test_images, mean, std_dev)
    _, _, normalized_val_data = normalize_data(val_images, mean, std_dev)

    # now let's have some fun and train the model
    best_model, best_hyperparams, best_accuracy = tune_model(normalized_train_data, train_labels, normalized_val_data, val_labels)

    print(f"Best Hyperparameters: kernel={best_hyperparams[0]}, tol={best_hyperparams[1]}, c_val={best_hyperparams[2]}")
    print(f"Best Validation Accuracy: {best_accuracy * 100}%")

    analyze_best_model(
        best_model,
        best_hyperparams,
        normalized_train_data, train_labels,
        normalized_val_data, val_labels,
        normalized_test_data, test_labels
    )


def main():
    # argument parser
    parser = argparse.ArgumentParser(description="Run or train a model and generate performance statistics.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-best", action="store_true", help="Run the model with the best hyperparameters.")
    group.add_argument("-train", action="store_true", help="Train the model and generate the asks")

    args = parser.parse_args()

    if args.best:
        run_best_model()
    elif args.train:
        train_and_evaluate()


if __name__ == "__main__":
    main()
