import torch
import torch.nn as nn
from torchvision import models

class CustomEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        print(base_model)

        for param in base_model.features.parameters():
            param.requires_grad = False

        self.features = base_model.features

        self.avgpool = base_model.avgpool

        self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1536 + 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 12)
        )

    def forward(self, x):
        img = x[0]
        bbox = x[1]
        #id = x[2] # apply as categorical instead of continuous varable

        x_center = bbox[:, 0] + 0.5 * bbox[:, 2]
        y_center = bbox[:, 1] + 0.5 * bbox[:, 3]
        height = bbox[:, 3]
        width = bbox[:, 2]

        img_features = self.features(img)
        img_features = self.avgpool(img_features)
        img_features = self.flatten(img_features)

        extra_features = [x_center, y_center, height, width]
        extra_features = torch.stack(extra_features, dim=1)

        features = torch.cat((img_features, extra_features), dim=1)

        return self.regressor(features)

CustomEfficientNet()






