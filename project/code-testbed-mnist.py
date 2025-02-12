import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_data(path):
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to PyTorch tensors
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    images, labels = [], []
    for imgs, lbls in data_loader:
        flattened = imgs.view(imgs.size(0), -1).numpy()  # Flatten images
        images.append(flattened)
        labels.extend(lbls.numpy())

    images = np.vstack(images)
    labels = np.array(labels)
    return images, labels


def normalize_data(data, mean=None, std_dev=None):
    if mean is None or std_dev is None:
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)

    std_dev[std_dev == 0] = 1e-8

    normalized_data = (data - mean) / std_dev
    return mean, std_dev, normalized_data


def train_model(training_images, training_labels):
    # Instantiate Logistic Regression model
    model = LogisticRegression(penalty=None, max_iter=1000, verbose=1)
    model.fit(training_images, training_labels)  # Train the model
    return model


def evaluate_model(model, test_images, test_labels):
    # Get predictions
    predictions = model.predict(test_images)
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Define paths to data
    train_data_path = "./mnist_data/train"
    test_data_path = "./mnist_data/test"

    # Load and preprocess data
    train_images, train_labels = get_data(train_data_path)
    test_images, test_labels = get_data(test_data_path)

    # Normalize data
    mean, std_dev, normalized_train_data = normalize_data(train_images)
    _, _, normalized_test_data = normalize_data(test_images, mean, std_dev)

    # Train the model
    model = train_model(normalized_train_data, train_labels)

    # Evaluate the model
    evaluate_model(model, normalized_test_data, test_labels)

    # # Train the model
    # model = train_model(train_images, train_labels)
    #
    # # Evaluate the model
    # evaluate_model(model, test_images, test_labels)
