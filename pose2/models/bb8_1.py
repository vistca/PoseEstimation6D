import torch.nn as nn
from torchvision.models import resnet18

class BB8Model_1(nn.Module):
    def __init__(self, num_objects=15):
        super().__init__()
        # We use ResNet18 as the backbone instead of VGG for greater efficiency
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # We remove the final fully connected layer

        # Head for predicting 2D points (8 corners * 2 coordinates)
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16)# 8 points * 2 coordinates
        )

        # Head for the symmetry classifier (4 ranges for approximately symmetrical objects)
        self.symmetry_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_objects * 4),
            nn.Sigmoid()
        )

        # Cache for 3D models (not used by BB8Model itself, but kept for potential future use)
        self.model_points_cache = {}

    def get_dimensions(self):
        return (224, 224)

    def forward(self, x):
        features = self.backbone(x)
        bbox_pred = self.bbox_head(features)
        symmetry_pred = self.symmetry_head(features)
        return bbox_pred, symmetry_pred
    

