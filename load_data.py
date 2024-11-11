import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToPILImage()
])

# train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

def generate_data():
    train_dataset = datasets.MNIST(root='data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=False, transform=transform)
    return train_dataset, test_dataset

if __name__ == "__main__":
    generate_data()