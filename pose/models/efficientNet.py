import torch
import torch.nn as nn
from torchvision import models

class CustomEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

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
        imgs = torch.cat([sample["rgb"] for sample in x], dim=0)
        bboxs = torch.cat([sample["bbox"] for sample in x], dim=0)

        #id = x[2] # apply as categorical instead of continuous varable

        x_center = bboxs[:, 0] + 0.5 * bboxs[:, 2]
        y_center = bboxs[:, 1] + 0.5 * bboxs[:, 3]
        height = bboxs[:, 3]
        width = bboxs[:, 2]

        img_features = self.features(imgs)
        img_features = self.avgpool(img_features)
        img_features = self.flatten(img_features)

        extra_features = [x_center, y_center, height, width]
        extra_features = torch.stack(extra_features, dim=1)

        features = torch.cat((img_features, extra_features), dim=1)

        return self.regressor(features)








