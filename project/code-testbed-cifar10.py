import os.path
import PIL
import torchvision.datasets
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.decomposition import PCA

def get_data(path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32 if necessary
        transforms.ToTensor(),  # Convert to PyTorch tensors
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_images = []
    train_labels = []

    dataset = datasets.ImageFolder(root=path, transform=transform)

    visualize_image(dataset[0])


    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for images, labels in train_loader:


        flattened = images.view(images.size(0), -1).numpy()  # Flatten each image
        train_images.append(flattened)
        train_labels.extend(labels.numpy())

    train_images = np.vstack(train_images)
    print(train_images.shape)
    train_labels = np.array(train_labels)

    return train_images, train_labels

def get_cifar10_data(train=True):
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # Resize images to 32x32 if necessary
        transforms.ToTensor(),  # Convert to PyTorch tensors
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_images = []
    train_labels = []

    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for images, labels in train_loader:
        flattened = images.view(images.size(0), -1).numpy()  # Flatten each image
        train_images.append(flattened)
        train_labels.extend(labels.numpy())

    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)

    return train_images, train_labels

def reduce_dimensions(training_data, pca=None):
    if pca is None:
        pca = PCA(n_components=10)

    train_images_pca = pca.fit_transform(training_data)

    # Explained variance ratio
    # print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

    return pca, train_images_pca

def normalize_data(training_data, mean=None, std_dev=None):
    if mean is None or std_dev is None:
        mean = np.mean(training_data, axis=0)
        std_dev = np.std(training_data, axis=0)

    normalized_data = (training_data - mean) / std_dev
    return mean, std_dev, normalized_data

class FNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=8),
            nn.ReLU(),
            # nn.Linear(in_features=4, out_features=8),
            # nn.ReLU(),
            nn.Linear(in_features=8, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=num_classes),
        )

    def forward(self, x):
        return self.fc(x)

def train_model(training_images, training_labels):

    model = FNN(input_dim=10, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data_pca = torch.utils.data.TensorDataset(
        torch.tensor(training_images, dtype=torch.float32),
        torch.tensor(training_labels, dtype=torch.long)
    )
    train_loader_pca = DataLoader(train_data_pca, batch_size=32, shuffle=True)

    # print(torch.cuda.is_available())
    # device = torch.device("cuda")
    #
    # model.to(device)

    # Training loop
    for epoch in range(100):  # 10 epochs
        for inputs, labels in train_loader_pca:
            # inputs.to(device)
            # labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in train_loader_pca:
                    # inputs.to(device)
                    # labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total * 100
            print(f"Training Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, test_images, test_labels):

    model.eval()  # Set to evaluation mode
    test_data = torch.utils.data.TensorDataset(
        torch.tensor(test_images, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader_pca = DataLoader(test_data, batch_size=32, shuffle=False)

    # Evaluate accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader_pca:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

import matplotlib.pyplot as plt
import numpy as np
def visualize_image(image):
    """Visualizes a single CIFAR-10 image.

    Args:
        image: A NumPy array representing the image.
    """

    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # let's first get all the data into memory?
    train_data_path = "./cifar10_data/automobile_train"
    test_data_path = "./cifar10_data/automobile_test"

    train_images, train_labels = get_data(train_data_path)
    test_images, test_labels = get_data(test_data_path)

    train_images, train_labels = get_cifar10_data(train=True)
    test_images, test_labels = get_cifar10_data(train=False)


    # Assuming you have the `train_images` array
    image_index = 10  # You can change this index to visualize different images
    image = train_images[image_index]

    # visualize_image(image)

    # # normalize data
    # mean, std_dev, normalized_data = normalize_data(train_images)
    # _, _, normalized_test_data = normalize_data(test_images, mean, std_dev)
    #
    # # print(normalized_data.shape)
    #
    # # Apply PCA
    # pca, train_images_pca = reduce_dimensions(normalized_data)
    # _, test_images_pca = reduce_dimensions(normalized_test_data, pca)
    #
    # # let's feed it to a NN and see how it goes
    # model = train_model(train_images_pca, train_labels)
    #
    # evaluate_model(model, test_images_pca, test_labels)