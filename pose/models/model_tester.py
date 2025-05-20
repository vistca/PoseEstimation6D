
import torch
from torchvision import transforms
from PIL import Image
from efficientNet import CustomEfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomEfficientNet().to(device)
model.eval()

# Create random image tensor (batch of 8 RGB images, 300x300)
img_tensor = torch.randn(8, 3, 300, 300).to(device)

# Create random bbox tensor (batch of 8, each with [x, y, w, h])
bbox_tensor = torch.rand(8, 4).to(device) * 300  # example scale

# Forward pass
with torch.no_grad():
    output = model((img_tensor, bbox_tensor))  # should return [8, 12]

print("Output shape:", output.shape)
print("Output:", output)
