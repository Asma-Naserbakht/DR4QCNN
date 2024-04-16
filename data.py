import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#### Train Data
def get_dataloader(args, train = True):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST data
    mnist_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    # Filter the dataset for classes 0 and 1
    class_0_and_1 = [0, 1]
    indices_selected = []
    for c in class_0_and_1:
        count = 0
        for i, (image, label) in enumerate(mnist_dataset):
            if label == c:
                indices_selected.append(i)
                count += 1
                if count == args.data_size:
                    break

    # Create a subset based on the selected indices
    subset_data = torch.utils.data.Subset(mnist_dataset, indices_selected)

    # Create a DataLoader for the subset data
    subset_loader = torch.utils.data.DataLoader(subset_data, batch_size=len(subset_data), shuffle=True)

    return subset_loader


def get_AE_train_data():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST data
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter the dataset for classes 0 and 1
    class_0_and_1 = [0, 1]
    indices_selected = [i for i, (image, label) in enumerate(mnist_dataset) if label in class_0_and_1]


    # Create a subset based on the selected indices
    subset_data = torch.utils.data.Subset(mnist_dataset, indices_selected)

    # Create a DataLoader for the subset data
    subset_loader = DataLoader(subset_data, batch_size=64, shuffle=True)

    return subset_loader