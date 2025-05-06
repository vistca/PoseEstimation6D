import torch.nn as nn
import torchvision
import yaml

class FasterRCNN():
    def __init__(self):
        with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

        # TODO: Look at the error UserWarning: Arguments other than a weight enum or `None` for 'weights' 
        # are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent 
        # to passing `weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. 
        # You can also use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` to get the most up-to-date weights.

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, config_dict['output_channels'])

        self.model = model

    def get_model(self):
        return self.model
    
