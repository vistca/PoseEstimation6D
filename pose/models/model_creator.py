from pose.models.efficientNet import CustomEfficientNet
from pose.models.resnet import CustomResNet50


def create_model(model_id):
    if model_id == "eff":
        print("Selected model: CustomEfficientNet")
        return CustomEfficientNet()
    elif model_id == "res":
        print("Selected model: CustomResNet50")
        return CustomResNet50()
    else:
        return ValueError("No such model exist!")




