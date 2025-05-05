import torch.nn as nn


class TestHead(nn.Module):

    def __init__(self, backbone_out_channels, nr_outputs):
        super(TestHead, self).__init__()
        
        self.head_layers = nn.Sequential([
            nn.Linear(backbone_out_channels, nr_outputs)
        ])
        
    def call(self, x):
        x =  self.head_layers(x)
        return x