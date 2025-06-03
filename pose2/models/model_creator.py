from pose2.models.CustomBB8 import CustomBB8Net

def create_model(model_id):
    if model_id == "res":
        print("Selected model: CustomBB8Net based on Resnet")
        return CustomBB8Net()
    else:
        return ValueError("No such model exist!")



