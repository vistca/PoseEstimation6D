import torch
import torch.nn as nn
from torchvision import models

class EfficientNet():
    def __init__(self):

        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 12)
        )

        self.model = model

    def get_model(self):
        return self.model




