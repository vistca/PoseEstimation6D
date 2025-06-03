
import torch.nn as nn
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomBB8Net(nn.Module):
    def __init__(self, num_objects=15):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        #for param in base_model.parameters():
        #    param.requires_grad = False

        self.features = nn.Sequential(
            *(list(base_model.children())[:-1])
        )
        
        self.dimensions = (224, 224)

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(2048 + 4 + 15),  # ResNet50 feature size = 2048
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 16)
        )

    def get_dimensions(self):
        return self.dimensions

    def forward(self, x):
    
        device = x[0]["rgb"].device
        imgs = torch.cat([sample["rgb"] for sample in x], dim=0)
        bbox = torch.cat([sample["bbox"] for sample in x], dim=0)
        obj_id = torch.cat([sample["obj_id"] for sample in x], dim=0)

        zero_based_id = obj_id - 1

        # Feature extraction
        img_features = self.features(imgs)
        #img_features = self.avgpool(img_features)
        img_features = self.flatten(img_features)

        # Bounding box and ID features concatinated with the image
        id_feature = F.one_hot(zero_based_id, num_classes=15).float().to(device)
    
        features = torch.cat((img_features, bbox, id_feature), dim=1)

        return self.regressor(features)
    


