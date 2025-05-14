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

        # Trainable layers -> 0 resulst in the backbone not being 
        # trainable at all, 5 is all layers are trainable
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=0)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, config_dict['output_channels'])
        
        # Disable backbone grads?
        #model.backbone.requires_grad_(False)

        # Enable grads for batchnorm maybe? Since that should be learned for our dataset

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
    
