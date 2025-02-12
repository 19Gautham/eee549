import torch
import torchvision
import torchvision.transforms as transforms
import os
import random
from PIL import Image


# Create directories for saving images
def create_dirs(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)


# Function to save images for specific classes
def save_images(dataset, labels, save_dir, class_names, num_samples):
    counts = {class_name: 0 for class_name in class_names.values()}
    total_needed = num_samples * len(class_names)

    # Ensure subdirectories exist
    for class_name in class_names.values():
        os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        if label in class_names:
            label_name = class_names[label]
            if counts[label_name] < num_samples:
                # img = transform(img)
                img.save(os.path.join(save_dir, label_name, f"{label_name}_{counts[label_name]}.png"))
                counts[label_name] += 1
            if sum(counts.values()) >= total_needed:
                break
    print(f"Saved {num_samples} images for each class in {save_dir}.")


# Function to save random samples for the "other" class (excluding cats and dogs)
def save_random_samples(images, save_dir, num_samples):
    random_samples = random.sample(images, num_samples)
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, label) in enumerate(random_samples):
        # img = transform(img)
        img.save(os.path.join(save_dir, f"other_{i}.png"))
    print(f"Saved {num_samples} images for 'other' class in {save_dir}.")


# Filter images for the "other" class by excluding specific labels (cats and dogs)
def filter_other_classes(dataset, exclude_labels):
    return [(img, label) for img, label in dataset if label not in exclude_labels]


# Define the transformation to apply to the images (convert to PIL)
transform = transforms.Compose([transforms.ToPILImage()])

# Directories to save the data
cifar10_dir = "cifar10_data"
cifar100_dir = "cifar100_data"
create_dirs(cifar10_dir, ['cats_train', 'dogs_train', 'cats_test', 'dogs_test'])
create_dirs(cifar100_dir, ['other_train', 'other_test'])

# --- CIFAR-10: Extract Cats and Dogs from Train and Test Sets ---
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Labels for cats and dogs in CIFAR-10
cifar10_labels = {3: "cats", 5: "dogs"}
num_samples = 3000

# Save cats and dogs images from training set
save_images(cifar10_train, cifar10_train.targets, os.path.join(cifar10_dir, 'cats_train'), cifar10_labels, num_samples)
save_images(cifar10_train, cifar10_train.targets, os.path.join(cifar10_dir, 'dogs_train'), cifar10_labels, num_samples)

# Save cats and dogs images from test set
save_images(cifar10_test, cifar10_test.targets, os.path.join(cifar10_dir, 'cats_test'), cifar10_labels, num_samples)
save_images(cifar10_test, cifar10_test.targets, os.path.join(cifar10_dir, 'dogs_test'), cifar10_labels, num_samples)

# --- CIFAR-100: Extract "Other" (Random images from classes excluding cats and dogs) ---
cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

# Get class labels and exclude cat and dog classes in CIFAR-100
exclude_labels = {}
cifar100_labels = cifar100_train.classes
exclude_indices = [cifar100_labels.index(label) for label in exclude_labels]
all_other_classes = [i for i in range(len(cifar100_labels)) if i not in exclude_indices]

# Filter out cat and dog images
other_train_images = filter_other_classes(cifar100_train, exclude_indices)
other_test_images = filter_other_classes(cifar100_test, exclude_indices)

# Randomly sample and save 200 images for the "other" class from training set
save_random_samples(other_train_images, os.path.join(cifar100_dir, 'other_train'), num_samples)

# Randomly sample and save 200 images for the "other" class from test set
save_random_samples(other_test_images, os.path.join(cifar100_dir, 'other_test'), num_samples)

print("Images successfully saved for CIFAR-10 and CIFAR-100 datasets.")
