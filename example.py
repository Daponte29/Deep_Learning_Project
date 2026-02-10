"""Quick example demonstrating the Deep Learning project"""
import torch
from src.models import SimpleCNN
from src.utils import get_device


def main():
    """Run a simple example"""
    print('='*60)
    print('Deep Learning Project - Quick Example')
    print('='*60)
    
    # Get device
    print('\n1. Device Setup')
    device = get_device()
    
    # Create model
    print('\n2. Model Creation')
    model = SimpleCNN(num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'   Model: SimpleCNN')
    print(f'   Parameters: {num_params:,}')
    
    # Create dummy input
    print('\n3. Sample Prediction')
    dummy_image = torch.randn(1, 1, 28, 28).to(device)
    print(f'   Input shape: {dummy_image.shape}')
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(dummy_image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities.max().item()
    
    print(f'   Predicted class: {predicted_class}')
    print(f'   Confidence: {confidence:.2%}')
    
    print('\n' + '='*60)
    print('Example completed successfully!')
    print('='*60)
    print('\nNext steps:')
    print('1. Run train.py to train the model on MNIST dataset')
    print('2. Run inference.py to make predictions with trained model')
    print('3. Run tests: python -m unittest tests.test_model')


if __name__ == '__main__':
    main()
