"""Training script for the Deep Learning model"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models import SimpleCNN
from src.data import get_mnist_loaders
from src.utils import get_device


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        float: Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    return test_loss, test_accuracy


def main():
    """Main training function"""
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    
    print('='*50)
    print('Deep Learning Project - PyTorch CNN Training')
    print('='*50)
    
    # Get device
    device = get_device()
    
    # Load data
    print('\nLoading MNIST dataset...')
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    
    # Initialize model
    print('\nInitializing model...')
    model = SimpleCNN(num_classes=10).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f'\nStarting training for {num_epochs} epochs...')
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')
    
    print('\n' + '='*50)
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')
    print('='*50)


if __name__ == '__main__':
    main()
