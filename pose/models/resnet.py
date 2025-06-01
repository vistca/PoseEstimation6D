import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.projections import directions_from_bboxs

class CustomResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.dimensions = (224,224)

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        #for param in base_model.parameters():
        #    param.requires_grad = False

        self.features = nn.Sequential(
            *(list(base_model.children())[:-1])
        )

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch, 2048, 1, 1)
        self.flatten = nn.Flatten()  # Flattens to (batch, 2048)

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(2048 + 4 + 15 + 8, 256),  # ResNet50 feature size = 2048
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 12)
        )

    def get_dimension(self):
        return self.dimensions

    def forward(self, x):
        imgs = torch.cat([sample["rgb"] for sample in x], dim=0)
        bbox = torch.cat([sample["bbox"] for sample in x], dim=0)
        obj_id = torch.cat([sample["obj_id"] for sample in x], dim=0)

        bbox_directions = directions_from_bboxs(bbox).float()
        zero_based_id = obj_id - 1

        # Feature extraction
        img_features = self.features(imgs)
        #img_features = self.avgpool(img_features)
        img_features = self.flatten(img_features)

        # Bounding box and ID features concatinated with the image
        id_feature = F.one_hot(zero_based_id, num_classes=15).float()
    
        features = torch.cat((img_features, bbox, id_feature, bbox_directions), dim=1)

        return self.regressor(features)

