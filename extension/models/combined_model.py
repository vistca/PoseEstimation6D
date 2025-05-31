import torch.nn as nn
import torch
from .depth_nn import DepthNN
from .resnet import CustomResNet50
from .rgb_nn import RgbNN

class CombinedModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rgb_model = RgbNN()
        self.depth_model = DepthNN()
        depth_out_channels = self.depth_model.get_output_channels()
        rgb_out_channels = self.rgb_model.get_output_channels()

        pose_in_channels = rgb_out_channels + depth_out_channels
        print("Pose in channels:", pose_in_channels)

        self.pose_model = CustomResNet50(pose_in_channels)
        self.global_model = None

    """
        Here we do the forward pass and pass the image to the different parts of the model
        This way we do not have to do it in the training loop and thus we can treat this
        as a single large model instead of several small ones
    """
    def forward(self, x):
        imgs = torch.cat([sample["rgb"] for sample in x], dim=0)
        depth = torch.cat([sample["depth"] for sample in x], dim=0).unsqueeze(1)
        bbox = torch.cat([sample["bbox"] for sample in x], dim=0)
        obj_id = torch.cat([sample["obj_id"] for sample in x], dim=0)
        

        rgb_output = self.rgb_model.forward(imgs)
        depth_output = self.depth_model.forward(depth)

        pose_input = torch.concat([rgb_output, depth_output], dim=1)

        if self.global_model: 
            global_features = self.global_model.forward(pose_input)
            pose_input = torch.concat([pose_input, global_features], dim=1)

        return self.pose_model(pose_input)  
    
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
        return (224,224)
