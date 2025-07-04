import torch.nn as nn
import torch

class RgbNN(nn.Module):
    
    def __init__(self, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        # These layers perserve the dimensions of the image as a result of padding
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channels,  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        )
        

    def forward(self, x):
        return self.layers(x)
    
    def get_output_channels(self):

        for i in range(1, len(self.layers)):
            if hasattr(self.layers[-i], "out_channels"):
                return self.layers[-i].out_channels

        return -1