from pose2.models.CustomBB8 import CustomBB8Net
from pose2.models.bb8_1 import BB8Model_1
from pose2.models.bb8_2 import BB8Model_2
from pose2.models.bb8_3 import BB8Model_3

def create_model(model_id):
    if model_id == "bb8_1":
        print("Selected the original model based on ResNet18")
        return BB8Model_1()
    elif model_id == "bb8_2":
        print("Selected model based on ResNet50, half frozen conv")
        return BB8Model_2()
    elif model_id == "bb8_3":
        print("Selected model based on ResNet50, no frozen conv")
        return BB8Model_3()
    else:
        return ValueError("No such model exist!")


