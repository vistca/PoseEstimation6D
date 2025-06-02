from pose2.models.bb8_1 import BB8Model_1
import torch
from PIL import Image, ImageDraw
import json
import numpy as np
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "best"

dir = "08"
nr = "230"
img_nr = "0" * (4 - len(nr))
img_nr = img_nr + nr

img_path = "./datasets/Linemod_preprocessed/data/" + dir + "/rgb/"
info_path = "./datasets/Linemod_preprocessed/data/" + dir + "/"
image_name = img_nr + ".png"
json_name = "gt.json"

model = BB8Model_1()

img = Image.open(img_path + image_name).convert("RGB")

pose_data = {}

with open(info_path + json_name, 'r') as f:
    pose_data = json.load(f)

# [x_left, y_top, x_width, y_height]
bbox = pose_data[nr][0]["obj_bb"] # bounding box

x_min, y_min, width, height = bbox
x_max = x_min + width
y_max = y_min + height
bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

margin = 0.2
crop_dim = (x_min - margin*width, y_min - margin*height, x_max + margin*width, y_max + margin*height)

crop = img.crop(crop_dim)

resize_format = model.get_dimensions()
crop = crop.resize(resize_format)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

checkpoint = torch.load(f"./checkpoints/{model_name}.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


input_img = transform(crop)

pred_points, _ = model(input_img.unsqueeze(0))
pred_points = pred_points.squeeze(0)

point_pairs = []

nr_points = pred_points.size()[0] // 2
img_side_size = model.get_dimensions()[0]


draw = ImageDraw.Draw(img)

for i in range(nr_points):
    point_x = pred_points[2*i].item() * width + x_min
    point_y = pred_points[2*i+1].item() * height + y_min
    
    point = (point_x, point_y)
    point_pairs.append(point)

    point_size = 3


edges = [(0,1), (0,3), (1,2), (2,3), (4,5), (4,7), (5,6), (6,7), (0,4), (1,5), (2,6), (3,7)]

for edge in edges:
    point1 = point_pairs[edge[0]]
    point2 = point_pairs[edge[1]]
    draw.line((point1, point2), fill="#A900DD", width=2)

img.show()

