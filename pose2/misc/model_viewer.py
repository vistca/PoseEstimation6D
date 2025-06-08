from pose2.models.model_creator import create_model

model = create_model("bb8_2")

print(model)
print("-"*25)

print(model.backbone.layer1)

