import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.projections import directions_from_bboxs

class GTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dimensions = (224,224)
        self.layer1 = nn.Linear(12, 12)

    def get_dimension(self):
        return self.dimensions

    def forward(self, x):

        t = torch.cat([sample["t"] for sample in x], dim=0)
        R = torch.cat([sample["R"] for sample in x], dim=0)

        features = torch.cat((t, R), dim=1)

        return self.layer1(features)

