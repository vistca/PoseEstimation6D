from pose2.models.CustomBB8 import CustomBB8Net
from pose2.models.bb8_2 import BB8Model_2

def create_model(model_id):
    if model_id == "res18":
        print("Selected old model: BB8Model_2 based on Resnet18")
        return BB8Model_2()
    else:
        return ValueError("No such model exist!")


