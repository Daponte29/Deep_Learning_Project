import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes=5, pretrained=True):
    """
    Factory function to get a model architecture.
    """
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vit_b_16':
        # Vision Transformer (ViT-B/16)
        # Note: Input size must be 224x224 for pretrained weights
        if pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        model = models.vit_b_16(weights=weights)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'efficientnet_b0':
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model
