"""MNIST dataset loader"""
import torch
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """
    Get MNIST train and test data loaders.
    
    Args:
        batch_size (int): Batch size for data loaders. Default is 64.
        data_dir (str): Directory to store/load MNIST data. Default is './data'.
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    return train_loader, test_loader
