import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

depth_path = "./datasets/Linemod_preprocessed/data/01/depth/0000.png"

# Load the depth image (unchanged to preserve 16-bit depth)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

cropped = depth_raw[0:200, 0:100]

resized = cv2.resize(cropped, (224, 224))  # e.g., (224, 224)

# Optional: Convert to meters
depth_m = cropped.astype(np.float32) / 1000.0

print(torch.tensor(depth_m))
print(torch.tensor(depth_m).unsqueeze(0).shape)

# Visualize
plt.imshow(depth_m, cmap='gray')
plt.colorbar(label='Depth (meters)')
plt.title('Depth Image')
plt.show()
