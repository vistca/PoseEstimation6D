from torchvision.models import resnet50, ResNet50_Weights
import yaml

class ResNet():

    def __init__(self):
        with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

    # Old weights with accuracy 76.130%
    model1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # New weights with accuracy 80.858%
    model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)