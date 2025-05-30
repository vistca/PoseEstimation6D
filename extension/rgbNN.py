import torch.nn as nn
import torch

class RgbNN(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential([
            nn.Conv2d(in_channels=1, out_channels=16),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128),
            nn.BatchNorm2d(),
            nn.ReLU(),
        ])
        

    def forward(self, x):
        return self.layers(x)
    


