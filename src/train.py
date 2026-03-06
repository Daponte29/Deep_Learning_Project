import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.dataset import CheXpertDataset, get_transforms
from src.models.net import get_model
from src.config import Config

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Calculate AUROC for each class
    auroc_per_class = []
    for i in range(all_labels.shape[1]):
        try:
            score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            score = 0.5
        auroc_per_class.append(score)
        
    avg_auroc = np.mean(auroc_per_class)
    
    return epoch_loss, avg_auroc, auroc_per_class

def main():
    parser = argparse.ArgumentParser(description='Train CheXpert Model')
    parser.add_argument('--csv_path', type=str, default='data/chexpert_small.csv', help='Path to CSV file')
    parser.add_argument('--img_dir', type=str, default='data/images/', help='Path to images directory') 
    parser.add_argument('--model', type=str, default='densenet121', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='output/', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Transform
    # 224x224 is standard for ViT-B/16 and EfficientNet-B0 pretrained on ImageNet
    # DenseNet121 typically uses 224 (though CheXpert originally used 320)
    transform_train = get_transforms(img_size=224, is_train=True)
    transform_valid = get_transforms(img_size=224, is_train=False)
    
    # Dataset
    # Split train/val - assuming args.csv_path is the "training set" csv provided by CheXpert
    # Ideally should use separate valid.csv
    
    # Read CSV once
    full_df = pd.read_csv(args.csv_path)
    # Shuffle
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    
    train_size = int(0.9 * len(full_df))
    valid_size = len(full_df) - train_size
    train_df = full_df.iloc[:train_size]
    valid_df = full_df.iloc[train_size:]
    
    # Save temp csvs to avoid rewriting Dataset class
    train_df.to_csv('train_temp.csv', index=False)
    valid_df.to_csv('valid_temp.csv', index=False)
    
    img_dir = args.img_dir
    
    train_dataset = CheXpertDataset('train_temp.csv', img_dir, transform=transform_train, policy='u-ones')
    valid_dataset = CheXpertDataset('valid_temp.csv', img_dir, transform=transform_valid, policy='u-ones')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = get_model(args.model, num_classes=5, pretrained=True)
    model = model.to(device)
    
    # Optimization
    # CheXpert uses Adam with specific betas
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    
    # Loss for Multi-label classification = BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
    best_auroc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_auroc, aurocs = validate(model, valid_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid AUROC: {valid_auroc:.4f}")
        print(f"AUROC per class: {aurocs}")
        
        # Save best model
        if valid_auroc > best_auroc:
            best_auroc = valid_auroc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{args.model}.pth"))
            print("Saved Best Model!")

if __name__ == "__main__":
    main()
