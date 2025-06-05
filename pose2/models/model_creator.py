from pose2.models.CustomBB8 import CustomBB8Net
from pose2.models.bb8_1 import BB8Model_1
from pose2.models.bb8_2 import BBM

def create_model(model_id):
    if model_id == "bb8_1":
        print("Selected the original model based on ResNet18")
        return BB8Model_1()
    elif model_id == "bb8_1":
        print("Selected model based on ResNet50")
    else:
        return ValueError("No such model exist!")


