import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, policy="u-ones", classes=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            policy (string): "u-ones" | "u-zeros" | "u-ignore"
            classes (list): List of pathologies to use as targets.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.policy = policy
        
        # Default 5 classes for competition
        if classes is None:
            self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        else:
            self.classes = classes

        # Pre-process labels based on policy
        self.process_labels()

    def process_labels(self):
        # Fill NaNs with 0 (negative)
        self.data[self.classes] = self.data[self.classes].fillna(0)
        
        if self.policy == "u-ones":
            self.data[self.classes] = self.data[self.classes].replace(-1, 1)
        elif self.policy == "u-zeros":
            self.data[self.classes] = self.data[self.classes].replace(-1, 0)
        # For u-ignore, we handle masking in the loss function, simplistic replacement here for now:
        # u-ignore typically means masking the loss but for simplicity we treat -1 as 0 or 1.
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0]) # "Path" column
        image = Image.open(img_path).convert('RGB')
        
        labels = self.data.iloc[idx][self.classes].values.astype('float32')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)

def get_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
