import torch.nn as nn
import torchvision
import yaml
from torchvision.models.detection import fasterrcnn_resnet50_fpn, \
fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn

class FasterRCNN():
    def __init__(self, trainable, version):
        with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

        # TODO: Look at the error UserWarning: Arguments other than a weight enum or `None` for 'weights' 
        # are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent 
        # to passing `weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. 
        # You can also use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` to get the most up-to-date weights.

        # Trainable layers -> 0 resulst in the backbone not being 
        # trainable at all, 5 is all layers are trainable
        if version == "resnet":
            model = fasterrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=trainable)
        elif version == "transform":
            model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=trainable)
        elif version == "mobilenet":
            model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', trainable_backbone_layers=trainable)
        elif version == "mobilenet_320":
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT', trainable_backbone_layers=trainable)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, config_dict['output_channels'])

        self.model = model

        count = 0
        params = 0
        for param in self.model.parameters():
            if param.requires_grad == True:
                count += 1

            params += 1
            
        print("Percent of grad req is: ", count/params)

    def get_model(self):
        return self.model