import torch.nn as nn
import torch

class DepthNN(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # Try to avrg-pool features down to 224x224?

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64,  kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        

    def forward(self, x):
        return self.layers(x)
    
    def get_output_channels(self):

        for i in range(1, len(self.layers)):
            if hasattr(self.layers[-i], "out_channels"):
                return self.layers[-i].out_channels

        return -1