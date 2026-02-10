"""Simple CNN model for image classification"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    
    Architecture:
    - Conv2d layer (1 -> 32 channels, 3x3 kernel)
    - MaxPool2d (2x2)
    - Conv2d layer (32 -> 64 channels, 3x3 kernel)
    - MaxPool2d (2x2)
    - Fully connected layers (64*7*7 -> 128 -> 10)
    
    Designed for MNIST-like datasets (28x28 grayscale images)
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize the SimpleCNN model.
        
        Args:
            num_classes (int): Number of output classes. Default is 10.
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
