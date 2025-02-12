import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


def visualize_two_classes():
    """
    Visualize one image from automobile and truck classes in the CIFAR-10 dataset.
    """
    # CIFAR-10 class names and their corresponding indices
    classes = ['automobile', 'truck']
    class_indices = [1, 9]  # Indices of automobile and truck in CIFAR-10

    # Define transform to convert tensor to numpy for plotting
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10 dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Create a figure with subplots
    fig, axes = plt.subplots(len(classes), 1, figsize=(8, 10))
    # fig.suptitle('CIFAR-10 Dataset - One Image per Class', fontsize=16)

    # Iterate through the dataset
    for img, label in dataset:
        # Check if the label matches one of our target classes
        if label in class_indices:
            # Find the index in our classes list
            class_idx = class_indices.index(label)

            # Convert tensor to numpy and transpose for matplotlib
            img_np = img.numpy().transpose(1, 2, 0)

            # Plot the image
            axes[class_idx].imshow(img_np)
            axes[class_idx].set_title(classes[class_idx])
            axes[class_idx].axis('off')

            # If we've found images for both classes, break
            if all(ax.get_images() for ax in axes):
                break

    plt.tight_layout()
    plt.show()


# Call the visualization function
visualize_two_classes()