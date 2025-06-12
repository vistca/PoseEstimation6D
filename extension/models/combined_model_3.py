import torch.nn as nn
import torch
from .depth_nn import DepthNN
from .rgb_nn import RgbNN
from pose2.models.model_creator import create_model

class CombinedModel3(nn.Module):

    def __init__(self, device, pose_name):
        super().__init__()

        self.pose_model = create_model(pose_name)
        self.dims = self.pose_model.get_dimension()

        
        try: 
            conv = self.pose_model.backbone.conv1
            out_features = conv.out_channels
            kernel_size = conv.kernel_size
            stride = conv.stride
            bias = conv.bias
            padding=conv.padding
            self.pose_model.backbone.conv1 = nn.Identity()
        except:
            raise("The model isn't compatible, name the pretrained model: backbone")
        
        self.pose_model = self.pose_model.to(device)

        split_model_out_channels = int(out_features/4)
        self.rgb_model = RgbNN(split_model_out_channels, kernel_size, stride, padding, bias).to(device)
        self.depth_model = DepthNN(split_model_out_channels, kernel_size, stride, padding, bias).to(device)
        
        self.global_model = nn.Sequential(
            nn.Conv2d(in_channels=split_model_out_channels*2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=3, stride=1, padding=1),
        ).to(device)

    """
        Here we do the forward pass and pass the image to the different parts of the model
        This way we do not have to do it in the training loop and thus we can treat this
        as a single large model instead of several small ones
    """
    def forward(self, x):
        imgs = x["rgb"]
        depth = x["depth"]

        rgb_output = self.rgb_model.forward(imgs)
        depth_output = self.depth_model.forward(depth)

        pose_input = torch.concat([rgb_output, depth_output], dim=1)
        global_features = self.global_model.forward(pose_input)
        pose_input = torch.concat([pose_input, global_features], dim=1)

        return self.pose_model({"rgb": pose_input})  
    
    """
        We need this since we want the parameters of all models to be present when
        we optimize, thus all models are optimized simultaniously. Can think about it
        as a single large model
    """
    def get_parameters(self):
        model_params = []
        models = [self.rgb_model, self.depth_model, self.pose_model]

        if self.global_model:
            models.append(self.global_model)

        for model in models:
            model_params.extend([p for p in model.parameters() if p.requires_grad])

        return model_params
    
    def get_dimensions(self):
        return self.dims
