import torch
import torchvision.models
import os
import PIL
from PIL import Image

def create_dirs(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

def save_images(dataset, labels, save_dir, class_names, num_samples):
    counter = {class_name: 0 for class_name in class_names.values()}
    total_needed = num_samples * len(class_names)

    # Ensure subdirectories exist
    for class_name in class_names.values():
        os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        if label in class_names:
            label_name = class_names[label]
            if counter[label_name] < num_samples:
                # img = transform(img)
                img.save(os.path.join(save_dir, label_name, f"{label_name}_{counter[label_name]}.png"))
                counter[label_name] += 1
            if sum(counter.values()) >= total_needed:
                break
    print(f"Saved {num_samples} images for each class in {save_dir}.")

cifar10_dir = "cifar10_data"
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

print(len(cifar10_train.targets))

for index, class_name in enumerate(cifar10_train.classes):
    print(index, class_name)

cifar10_labels = {1: "automobiles", 9: "truck"}
num_samples = 100

if not os.path.exists('./cifar10_data/automobile_test'):
    print('Saving images as the data is not extracted at present')
    save_images(cifar10_train, cifar10_train.targets, os.path.join(cifar10_dir, 'automobile_test'), cifar10_labels, num_samples)

# img = Image.open('./cifar10_data/automobile_train/automobiles/automobiles_0.png')
# print(img.size)
