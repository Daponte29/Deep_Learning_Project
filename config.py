"""Configuration settings for the Deep Learning project"""


class Config:
    """Configuration class for training parameters"""
    
    # Data settings
    BATCH_SIZE = 64
    DATA_DIR = './data'
    NUM_WORKERS = 2
    
    # Model settings
    NUM_CLASSES = 10
    
    # Training settings
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    
    # Paths
    MODEL_SAVE_PATH = 'checkpoints/best_model.pth'
    
    # Device
    DEVICE = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
