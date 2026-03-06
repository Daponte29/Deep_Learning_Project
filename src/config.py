import os

class Config:
    # Paths
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    VALID_CSV = os.path.join(DATA_DIR, "valid.csv")
    OUTPUT_DIR = "./output"
    
    # Model
    MODEL_NAME = "densenet121"  # "resnet50", "vit_b_16"
    NUM_CLASSES = 14
    PRETRAINED = True
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    SEED = 42
    img_size = 224
    
    # Data Processing
    U_ZEROS = False  # If True, map -1 to 0; else map -1 to 1 (U-Ones)
