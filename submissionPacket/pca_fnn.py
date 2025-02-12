import argparse
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import optuna
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import itertools

import os.path
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score

def get_cifar10_data(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_images = []
    train_labels = []

    required_labels = {1: "automobiles", 9: "truck"}

    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    for image, label in dataset:
        if label in required_labels.keys():
          # Map "1" to "0" and "9" to "1"
          label = 0 if label == 1 else 1
          flattened = image.view(-1).numpy()
          train_images.append(flattened)
          train_labels.append(label)

    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    return train_images, train_labels

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(FNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def train_and_plot_accuracy(
    model, train_loader, test_loader, criterion, optimizer, device, epochs=50
):
    train_accuracies = []
    test_accuracies = []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_accuracy = evaluate_accuracy(model, train_loader, device)
        train_accuracies.append(train_accuracy)

        test_accuracy = evaluate_accuracy(model, test_loader, device)
        test_accuracies.append(test_accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Test Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    return model

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


def evaluate_model(model, test_images, test_labels):
    model.eval()
    test_data = TensorDataset(
        torch.tensor(test_images, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    return accuracy

def objective(trial, train_images, train_labels, test_images, test_labels):

    n_components = trial.suggest_int("n_components", 5, 200)
    hidden_size1 = trial.suggest_int("hidden_size1", 4, 256)
    hidden_size2 = trial.suggest_int("hidden_size2", 4, 256)
    hidden_size3 = trial.suggest_int("hidden_size3", 4, 128)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    # Apply PCA
    pca = PCA(n_components=n_components)
    train_images_pca = pca.fit_transform(train_images)
    test_images_pca = pca.transform(test_images)

    train_data = TensorDataset(
        torch.tensor(train_images_pca, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model = FNN(input_dim=n_components, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, criterion, optimizer, device, epochs=100)

    accuracy = evaluate_model(model, test_images_pca, test_labels)
    return accuracy


def tune_hyperparameter(training_images, training_labels, test_images, test_labels):
    learning_rates = [0.0001, 0.0003, 0.0006, 0.001, 0.002137129103186701, 0.005, 0.007, 0.01]
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    for lr in learning_rates:
        print(f"\nTraining with Learning Rate: {lr}")
        model, best_train_acc = train_model_for_tuning(
            training_images, training_labels, lr, epochs=30
        )

        # Get test accuracy for the trained model
        test_accuracy = evaluate_model(model, test_images, test_labels)

        # Store the accuracies
        train_accuracies.append(best_train_acc)
        test_accuracies.append(test_accuracy)

    print(train_accuracies)
    print(test_accuracies)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, train_accuracies, marker='o', label='Training Accuracy')
    # plt.plot(learning_rates, val_accuracies, marker='o', label='Validation Accuracy')
    plt.plot(learning_rates, test_accuracies, marker='o', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Learning Rate")
    plt.legend()
    plt.grid()
    plt.show()

    return train_accuracies, val_accuracies, test_accuracies


def plot_roc_curve(model, test_images, test_labels):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(test_images_tensor)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class "1"

    # Calculate ROC and AUROC
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    auroc = roc_auc_score(test_labels, probabilities)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"AUROC: {auroc:.2f}")


def per_class_accuracy(model, test_images, test_labels, label_names):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.tensor(test_images, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(test_data)
        _, predictions = torch.max(outputs, 1)

    # Calculate per-class accuracy
    per_class_acc = {}
    for label in np.unique(test_labels.cpu().numpy()):
        indices = (test_labels == label)
        class_acc = accuracy_score(test_labels[indices].cpu(), predictions[indices].cpu())
        per_class_acc[label_names[label]] = class_acc

    print("Per-Class Accuracy:")
    for label, acc in per_class_acc.items():
        print(f"{label}: {acc * 100:.2f}%")


def train_model_for_tuning(training_images, training_labels, lr, epochs=30):
    # Initialize the model
    model = FNN(112, 202, 165, 46, 2)
    # model = FNN(48, 243, 180, 75, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data loaders
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(training_images, dtype=torch.float32),
        torch.tensor(training_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # val_data = torch.utils.data.TensorDataset(
    #     torch.tensor(val_images, dtype=torch.float32),
    #     torch.tensor(val_labels, dtype=torch.long)
    # )
    # val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track the best accuracies
    best_train_accuracy = 0
    best_val_accuracy = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:

            train_accuracy = evaluate_accuracy(model, train_loader, device)
            # val_accuracy = evaluate_accuracy(model, val_loader, device)

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                # best_val_accuracy = val_accuracy

    return model, best_train_accuracy


def run_best_model():

    train_images, train_labels = get_cifar10_data(train=True)
    test_images, test_labels = get_cifar10_data(train=False)

    # Run Optuna study
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50)

    # # Output best parameters and accuracy
    # print("Best parameters:", study.best_params)
    # print("Best test accuracy:", study.best_value)

    best_params = {'n_components': 112, 'hidden_size1': 202, 'hidden_size2': 165, 'hidden_size3': 46,
                   'lr': 0.002137129103186701}

    # Train the final model with the best parameters
    # best_params = study.best_params
    pca = PCA(n_components=best_params["n_components"])
    train_images_pca = pca.fit_transform(train_images)
    test_images_pca = pca.transform(test_images)

    final_model = FNN(
        input_dim=best_params["n_components"],
        hidden_size1=best_params["hidden_size1"],
        hidden_size2=best_params["hidden_size2"],
        hidden_size3=best_params["hidden_size3"],
        num_classes=2,
    )

    train_data = TensorDataset(
        torch.tensor(train_images_pca, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()

    test_data = TensorDataset(
        torch.tensor(test_images_pca, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(final_model, train_loader, criterion, optimizer, device, epochs=100)

    final_accuracy = evaluate_model(model, test_images_pca, test_labels)
    print(f"Final test accuracy with best parameters: {final_accuracy:.2f}%")

    plot_roc_curve(model, test_images_pca, test_labels)

    label_names = {0: "Automobile", 1: "Truck"}  # Update with your label mapping

    # per class accuracy
    per_class_accuracy(model, test_images_pca, test_labels, label_names)

    train_accuracies, val_accuracies, test_accuracies = tune_hyperparameter(
        train_images_pca, train_labels, test_images_pca, test_labels
    )

    train_and_plot_accuracy(final_model, train_loader, test_loader, criterion, optimizer, device, epochs=100)


def train_and_evaluate():

    train_images, train_labels = get_cifar10_data(train=True)
    test_images, test_labels = get_cifar10_data(train=False)

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # # Output best parameters and accuracy
    print("Best parameters:", study.best_params)
    print("Best test accuracy:", study.best_value)

    # best_params = study.best_params

    # best_params = {'n_components': 112, 'hidden_size1': 202, 'hidden_size2': 165, 'hidden_size3': 46,
    #                'lr': 0.002137129103186701}

    # Train the final model with the best parameters
    best_params = study.best_params
    pca = PCA(n_components=best_params["n_components"])
    train_images_pca = pca.fit_transform(train_images)
    test_images_pca = pca.transform(test_images)

    final_model = FNN(
        input_dim=best_params["n_components"],
        hidden_size1=best_params["hidden_size1"],
        hidden_size2=best_params["hidden_size2"],
        hidden_size3=best_params["hidden_size3"],
        num_classes=2,
    )

    train_data = TensorDataset(
        torch.tensor(train_images_pca, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()

    test_data = TensorDataset(
        torch.tensor(test_images_pca, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(final_model, train_loader, criterion, optimizer, device, epochs=100)

    final_accuracy = evaluate_model(model, test_images_pca, test_labels)
    print(f"Final test accuracy with best parameters: {final_accuracy:.2f}%")

    plot_roc_curve(model, test_images_pca, test_labels)

    label_names = {0: "Automobile", 1: "Truck"}  # Update with your label mapping

    # per class accuracy
    per_class_accuracy(model, test_images_pca, test_labels, label_names)

    train_accuracies, val_accuracies, test_accuracies = tune_hyperparameter(
        train_images_pca, train_labels, test_images_pca, test_labels
    )

    train_and_plot_accuracy(final_model, train_loader, test_loader, criterion, optimizer, device, epochs=100)


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