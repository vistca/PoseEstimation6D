from backbones.test_backbone import TestBackbone
from heads.test_head import TestHead
import yaml
import torch.nn as nn


class ModelLoader(nn.Module):

    def __init__(self, headname, backbone):
        super(ModelLoader, self).__init__()
        
        with open('config/config.yaml') as f:
            config = yaml.safe_load(f)

        self.backbone = TestBackbone(config['input_channels'])
        backbone_out_channels = self.backbone.get_out_channels()
        self.head = TestHead(backbone_out_channels, config['output_channels'])
    
    def call(self, x):
        x = self.backbone.call(x)
        x = self.head.call(x)
        return x