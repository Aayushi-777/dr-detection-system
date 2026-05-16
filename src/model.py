import torch
import torch.nn as nn
import timm

class EfficientNet(nn.Module):
    """
    High-accuracy DR model using EfficientNet-B3
    """
    def __init__(self, num_classes=5, pretrained=True, dropout=0.3):
        super().__init__()

        self.backbone=timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        self.feature_dim=self.backbone.num_features

        self.attention=nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.classifier=nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features=self.backbone.forward_features(x)

        attn=self.attention(features)
        features=features*attn
        features=features.mean(dim=(2, 3))

        out=self.classifier(features)
        return out
    
def build_model(num_classes=5, device='cpu'):
    model=EfficientNet(num_classes=num_classes)
    return model.to(device)
    
def get_target_layer(model):
    return model.backbone.blocks[-1][-1].conv_pwl