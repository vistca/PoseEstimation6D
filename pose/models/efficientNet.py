import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CustomEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dimensions = (300, 300)

        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        for param in base_model.features.parameters():
            param.requires_grad = False

        self.features = base_model.features

        self.avgpool = base_model.avgpool

        self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1536 + 4 + 15, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 12)
        )


    def get_dimension(self):
        return self.dimensions


    def forward(self, x):
        
        imgs = torch.cat([sample["rgb"] for sample in x], dim=0)
        bboxs = torch.cat([sample["bbox"] for sample in x], dim=0)
        obj_id = torch.cat([sample["obj_id"] for sample in x], dim=0)
        zero_based_id = obj_id - 1

        x_center = bboxs[:, 0] + 0.5 * bboxs[:, 2]
        y_center = bboxs[:, 1] + 0.5 * bboxs[:, 3]
        height = bboxs[:, 3]
        width = bboxs[:, 2]

        img_features = self.features(imgs)
        img_features = self.avgpool(img_features)
        img_features = self.flatten(img_features)

        bbox_features = [x_center, y_center, height, width]
        bbox_features = torch.stack(bbox_features, dim=1)

        id_feature = F.one_hot(zero_based_id, num_classes=15).float()
        
        features = torch.cat((img_features, bbox_features, id_feature), dim=1)

        return self.regressor(features)








