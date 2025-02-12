import torchvision.models
import os

def create_dirs(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

def save_images(dataset, save_dir, class_names, num_samples):
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

def collect_data(train=True, num_samples = 100):

    mnist_dir = "mnist_data"
    mnist_dataset = None
    if train:
        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    else:
        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    # print(len(mnist_train.targets))
    #
    # for index, class_name in enumerate(mnist_train.classes):
    #     print(f"Index: {index}, Class: {class_name}")

    mnist_labels = {5: "5 - five", 6: "6 - six"}

    train_type = "train" if train else "test"

    if not os.path.exists(f'./mnist_data/{train_type}'):
        print('Saving images as the data is not extracted at present')
        save_images(mnist_dataset, os.path.join(mnist_dir, train_type), mnist_labels, num_samples)

    # img = Image.open('./cifar10_data/automobile_train/automobiles/automobiles_0.png')
    # print(img.size)

if __name__ == "__main__":
    collect_data(train=False, num_samples=100)