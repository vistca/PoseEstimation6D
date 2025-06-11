import torch.nn as nn
from torchvision.models import resnet50

class BB8Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        # We use ResNet18 as the backbone instead of VGG for greater efficiency
        self.backbone = resnet50(weights="DEFAULT")
        self.in_channels = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # We remove the final fully connected layer

        for param in self.backbone.layer1.parameters():
            param.requires_grad = False

        for param in self.backbone.layer2.parameters():
            param.requires_grad = False

        # Head for predicting 2D points (8 corners * 2 coordinates)
        self.bbox_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.in_channels, 512),  # ResNet50 feature size = 2048
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 16) # 8 points * 2 coordinates
        )

        # Cache for 3D models (not used by BB8Model itself, but kept for potential future use)
        # self.model_points_cache = {}

    def get_dimension(self):
        return (224, 224)

    def forward(self, inputs):
        images = inputs["rgb"]
        features = self.backbone(images)
        bbox_pred = self.bbox_head(features)
        return bbox_pred

