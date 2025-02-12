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

import matplotlib.pyplot as plt

import argparse

def get_cifar10_data(train=True):
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # Resize images to 32x32 if necessary
        transforms.ToTensor(),  # Convert to PyTorch tensors
    ])

    train_images = []
    train_labels = []

    required_labels = {1: "automobiles", 9: "truck"}

    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for image, label in dataset:
        if label in required_labels.keys():
          label = 0 if label == 1 else 1  # Mapping "1" to "0" and "9" to "1"
          train_images.append(image)
          train_labels.append(label)

    train_images = torch.stack(train_images).numpy()
    train_labels = np.array(train_labels)
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")


    return train_images, train_labels

class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # just some experimentation with dilated filters - don't bother
        # self.cnn = nn.Sequential(
        #     # 32 +4 -5 + 1 = 32
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2, dilation=2),
        #     nn.ReLU(),
        #     # ((32-2)/2) + 1 = 16
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # 16 +2 -3 + 1 = 16
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2),
        #     nn.ReLU(),
        #     # ((16-2)/2) + 1 = 8
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # 8 +2 -3 + 1 = 8
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2),
        #     nn.ReLU(),
        #     # ((8-2)/2) + 1 = 4
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        self.cnn = nn.Sequential(
            # 32 +4 -5 + 1 = 32
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # ((32-2)/2) + 1 = 16
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16 +2 -3 + 1 = 16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # ((16-2)/2) + 1 = 8
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8 +2 -3 + 1 = 8
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # ((8-2)/2) + 1 = 4
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

def train_model(training_images, training_labels, test_images_pca, test_labels):

    model = CNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data_pca = torch.utils.data.TensorDataset(
        torch.tensor(training_images, dtype=torch.float32),
        torch.tensor(training_labels, dtype=torch.long)
    )
    train_loader_pca = DataLoader(train_data_pca, batch_size=32, shuffle=True)

    # print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 100

    train_acc_list = []
    test_acc_list = []
    epochs_list = []

    # Training loop
    for epoch in range(epochs):  # 10 epochs
        for inputs, labels in train_loader_pca:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            epochs_list.append(epoch + 1)
            with torch.no_grad():
                for inputs, labels in train_loader_pca:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total * 100
            print(f"Training Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")
            train_acc_list.append(accuracy)
            test_acc_list.append(evaluate_model(model, test_images_pca, test_labels))

    # # Plot Bias-Variance Tradeoff Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_list, train_acc_list, label="Training Accuracy")
    plt.plot(epochs_list, test_acc_list, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, test_images, test_labels):

    model.eval()  # Set to evaluation mode
    test_data = torch.utils.data.TensorDataset(
        torch.tensor(test_images, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader_pca = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader_pca:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return test_accuracy

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import numpy as np

def per_class_accuracy(model, test_images, test_labels, label_names):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.tensor(test_images, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(test_data)
        _, predictions = torch.max(outputs, 1)

    per_class_acc = {}
    for label in np.unique(test_labels.cpu().numpy()):
        indices = (test_labels == label)
        class_acc = accuracy_score(test_labels[indices].cpu(), predictions[indices].cpu())
        per_class_acc[label_names[label]] = class_acc * 100

    print("Per-Class Accuracy:")
    for label, acc in per_class_acc.items():
        print(f"{label}: {acc:.2f}%")

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

def tune_hyperparameter(training_images, training_labels, test_images, test_labels):
    learning_rates = [0.0001, 0.0003, 0.0006, 0.001, 0.002137129103186701, 0.005, 0.007, 0.01]
    train_accuracies = []
    test_accuracies = []

    for lr in learning_rates:
        print(f"\nTraining with Learning Rate: {lr}")
        model = CNN(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_data = torch.utils.data.TensorDataset(
            torch.tensor(training_images, dtype=torch.float32),
            torch.tensor(training_labels, dtype=torch.long)
        )
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Train the model for 10 epochs
        for epoch in range(10):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate accuracies
        train_accuracy = evaluate_model(model, training_images, training_labels)
        test_accuracy = evaluate_model(model, test_images, test_labels)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(learning_rates, test_accuracies, marker='o', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Learning Rate")
    plt.legend()
    plt.grid()
    plt.show()

    print("Learning Rate Tuning Complete")
    return train_accuracies, test_accuracies

def run_best_model():

    # let's first get all the data into memory?
    train_images, train_labels = get_cifar10_data(train=True)
    test_images, test_labels = get_cifar10_data(train=False)

    label_names = {0: "Automobile", 1: "Truck"}

    # let's feed it to a NN and see how it goes
    model = train_model(train_images, train_labels, test_images, test_labels)
    # evaluate_model(model, test_images_pca, test_labels)

    # Evaluate model
    evaluate_model(model, test_images, test_labels)

    # Per-Class Accuracy
    per_class_accuracy(model, test_images, test_labels, label_names)

    # ROC and AUROC
    plot_roc_curve(model, test_images, test_labels)

    # Hyperparameter tuning
    tune_hyperparameter(train_images, train_labels, test_images, test_labels)

def train_and_evaluate():

    train_images, train_labels = get_cifar10_data(train=True)
    test_images, test_labels = get_cifar10_data(train=False)

    label_names = {0: "Automobile", 1: "Truck"}

    # let's feed it to a NN and see how it goes
    model = train_model(train_images, train_labels, test_images, test_labels)
    # evaluate_model(model, test_images_pca, test_labels)

    # Evaluate model
    evaluate_model(model, test_images, test_labels)

    # Per-Class Accuracy
    per_class_accuracy(model, test_images, test_labels, label_names)

    # ROC and AUROC
    plot_roc_curve(model, test_images, test_labels)

    # Hyperparameter tuning
    tune_hyperparameter(train_images, train_labels, test_images, test_labels)

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