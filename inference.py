"""Inference script for making predictions with the trained model"""
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from src.models import SimpleCNN
from src.utils import get_device


def load_model(model_path, num_classes=10):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model weights
        num_classes (int): Number of output classes
    
    Returns:
        model: Loaded PyTorch model
    """
    device = get_device()
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def predict_sample(model, image_tensor, device):
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained model
        image_tensor: Image tensor of shape (1, 28, 28)
        device: Device to run inference on
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()


def visualize_predictions(model, test_dataset, device, num_samples=10):
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run inference on
        num_samples (int): Number of samples to visualize
    """
    # Randomly select samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        
        # Make prediction
        predicted_label, confidence = predict_sample(model, image, device)
        
        # Display image
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].axis('off')
        
        # Set title with prediction
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}',
            color=color
        )
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print('Predictions saved to predictions.png')
    plt.close()


def main():
    """Main inference function"""
    print('='*50)
    print('Deep Learning Project - Model Inference')
    print('='*50)
    
    # Load model
    model_path = 'checkpoints/best_model.pth'
    print(f'\nLoading model from {model_path}...')
    
    try:
        model, device = load_model(model_path)
        print('Model loaded successfully!')
    except FileNotFoundError:
        print(f'Error: Model file {model_path} not found.')
        print('Please train the model first using train.py')
        return
    
    # Load test dataset
    print('\nLoading MNIST test dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Visualize predictions
    print('\nGenerating predictions visualization...')
    visualize_predictions(model, test_dataset, device, num_samples=10)
    
    # Interactive prediction
    print('\n' + '='*50)
    print('Model ready for inference!')
    print('You can use the predict_sample() function to make predictions')
    print('='*50)


if __name__ == '__main__':
    main()
