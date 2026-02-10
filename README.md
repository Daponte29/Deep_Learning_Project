# Deep Learning Project

A Deep Learning project using PyTorch for image classification tasks.

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It serves as a foundation for more complex deep learning projects.

## Features

- Simple CNN architecture for image classification
- Modular code structure with separate modules for models, data loading, and utilities
- Training script with progress tracking
- Support for CUDA, MPS (Apple Silicon), and CPU devices
- Easy-to-configure hyperparameters

## Project Structure

```
Deep_Learning_Project/
├── src/
│   ├── __init__.py
│   ├── models/          # Neural network models
│   │   ├── __init__.py
│   │   └── simple_cnn.py
│   ├── data/            # Data loading utilities
│   │   ├── __init__.py
│   │   └── mnist_loader.py
│   └── utils/           # Utility functions
│       ├── __init__.py
│       └── device.py
├── tests/               # Unit tests
│   ├── __init__.py
│   └── test_model.py
├── train.py             # Training script
├── inference.py         # Inference/prediction script
├── example.py           # Quick example demonstration
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Daponte29/Deep_Learning_Project.git
cd Deep_Learning_Project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script:
```bash
python train.py
```

The script will:
- Automatically download the MNIST dataset
- Train a CNN model for 5 epochs
- Display training progress and accuracy
- Save the best model to `best_model.pth`

### Model Architecture

The SimpleCNN model consists of:
- 2 Convolutional layers (with ReLU activation)
- 2 Max pooling layers
- 2 Fully connected layers
- Dropout for regularization

Input: 28x28 grayscale images  
Output: 10 classes (digits 0-9)

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- torchvision 0.15.0+
- numpy 1.24.0+
- matplotlib 3.7.0+
- tqdm 4.65.0+

See `requirements.txt` for the complete list of dependencies.

## Configuration

You can modify training parameters in `config.py`:
- Batch size
- Learning rate
- Number of epochs
- Data directory
- And more...

## Results

The model typically achieves:
- Training accuracy: ~99%
- Test accuracy: ~98%

(After 5 epochs on the MNIST dataset)

## Future Enhancements

Potential improvements and extensions:
- Add support for custom datasets
- Implement more advanced architectures (ResNet, VGG, etc.)
- Add data augmentation techniques
- Include visualization tools for model predictions
- Add model evaluation and inference scripts
- Implement learning rate scheduling
- Add TensorBoard logging

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Contact

For questions or suggestions, please open an issue on GitHub.