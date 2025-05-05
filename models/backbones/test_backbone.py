import torch.nn as nn


class TestBackbone(nn.Module):

    def __init__(self, in_channels):
        super(TestBackbone, self).__init__()
        self.out_channels = 30
        self.head_layers = nn.Sequential([
            nn.Conv2d(in_channels=in_channels, out_channels=20, stride=2, padding=1),
            nn.Conv2d(in_channels=20, out_channels=self.out_channels, stride=1, padding=1)
        ])
    
    def get_out_channels(self):
        return self.out_channels

    def call(self, x):
        x = self.head_layers(x)
        return x