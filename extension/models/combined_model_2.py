import torch.nn as nn
import torch
from .depth_nn import DepthNN
from .rgb_nn import RgbNN
from pose2.models.model_creator import create_model

class CombinedModel2(nn.Module):

    def __init__(self, device, pose_name):
        super().__init__()

        self.rgb_model = create_model(pose_name)
        self.depth_model = create_model(pose_name)
        self.dims = self.rgb_model.get_dimension()

        # Modifying the model to handel an imput with one channel for the depth
        in_channels = 1
        conv = self.depth_model.backbone.conv1
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        bias = conv.bias
        padding=conv.padding
        self.depth_model.backbone.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)

        # Removes the previous bbox head
        self.rgb_model.bbox_head = nn.Identity()
        self.depth_model.bbox_head = nn.Identity()

        self.fc_input_size = self.rgb_model.in_channels + self.depth_model.in_channels

        # Head for predicting 2D points (8 corners * 2 coordinates)
        self.bbox_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.fc_input_size, 512),  # ResNet50 feature size = 2048
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 16) # 8 points * 2 coordinates
        ).to(device)

        self.depth_model = self.depth_model.to(device)
        self.rgb_model = self.rgb_model.to(device)

        self.global_model = False

    """
        Here we do the forward pass and pass the image to the different parts of the model
        This way we do not have to do it in the training loop and thus we can treat this
        as a single large model instead of several small ones
    """
    def forward(self, x):
        imgs = x["rgb"]
        depth = x["depth"]

        # The pose predictor models from phase 3 takes in data labeled rgb
        rgb_output = self.rgb_model.forward({"rgb": imgs})
        depth_output = self.depth_model.forward({"rgb": depth})

        pose_input = torch.concat([rgb_output, depth_output], dim=1)

        if self.global_model: 
            global_features = self.global_model.forward(pose_input)
            pose_input = torch.concat([pose_input, global_features], dim=1)

        return self.bbox_head(pose_input)  
    
    """
        We need this since we want the parameters of all models to be present when
        we optimize, thus all models are optimized simultaniously. Can think about it
        as a single large model
    """
    def get_parameters(self):
        model_params = []
        models = [self.rgb_model, self.depth_model]

        if self.global_model:
            models.append(self.global_model)

        for model in models:
            model_params.extend([p for p in model.parameters() if p.requires_grad])

        return model_params
    
    def get_dimensions(self):
        return self.dims


