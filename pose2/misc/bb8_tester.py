from pose2.models.bb8_1 import BB8Model_1
import torch
from PIL import Image, ImageDraw
import json
import numpy as np
from torchvision import transforms
import cv2
from plyfile import PlyData
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "best"

id = "15"
dir = (2-len(id)) * "0" + id
nr = "230"
img_nr = "0" * (4 - len(nr))
img_nr = img_nr + nr

model_file_path = "./datasets/Linemod_preprocessed/models/models_info.json"
img_path = "./datasets/Linemod_preprocessed/data/" + dir + "/rgb/"
info_path = "./datasets/Linemod_preprocessed/data/" + dir + "/"
image_name = img_nr + ".png"
json_name = "gt.json"

model = BB8Model_1()

img = Image.open(img_path + image_name).convert("RGB")

pose_data = {}

with open(info_path + json_name, 'r') as f:
    pose_data = json.load(f)

with open(model_file_path, 'r') as f:
    models_info = json.load(f) 

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

checkpoint = torch.load(f"./pose2/checkpoints/{model_name}.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


input_img = transform(crop)

pred_points, _ = model(input_img.unsqueeze(0))
pred_points = pred_points.squeeze(0)






model_points_2d = np.empty((8,2))
nr_points = pred_points.size()[0] // 2
img_side_size = model.get_dimensions()[0]


draw = ImageDraw.Draw(img)

for i in range(nr_points):
    point_x = pred_points[2*i].item() * width + x_min
    point_y = pred_points[2*i+1].item() * height + y_min
    model_points_2d[i] = point_x, point_y

edges = [(0,1), (0,3), (1,2), (2,3), (4,5), (4,7), (5,6), (6,7), (0,4), (1,5), (2,6), (3,7)]

for edge in edges:
    x1, y1 = model_points_2d[edge[0]]
    x2, y2 = model_points_2d[edge[1]]
    draw.line(((x1, y1), (x2, y2)), fill="#FF5E00", width=2)

img.show()

info = models_info[id]
x_size = info['size_x']
y_size = info['size_y']
z_size = info['size_z']

half_x = x_size / 2
half_y = y_size / 2
half_z = z_size / 2

model_points_3d = np.array([
    [-half_x, -half_y, -half_z],
    [half_x, -half_y, -half_z],
    [half_x, half_y, -half_z],
    [-half_x, half_y, -half_z],
    [-half_x, -half_y, half_z],
    [half_x, -half_y, half_z],
    [half_x, half_y, half_z],
    [-half_x, half_y, half_z]
], dtype=np.float32)

camera_matrix = np.array([
    [572.4114, 0.0, 325.2611],
    [0.0, 573.57043, 242.04899],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

start = time.perf_counter()

success, pred_rvec, pred_pos = cv2.solvePnP(
    objectPoints=model_points_3d,
    imagePoints=model_points_2d,
    cameraMatrix=camera_matrix,
    distCoeffs=np.zeros(5),
    flags=cv2.SOLVEPNP_ITERATIVE
)


pred_rot_matrix, _ = cv2.Rodrigues(pred_rvec)

end = time.perf_counter()

print(f"Time to PnP: {int(1000*(end-start))}ms")

pred_t = torch.tensor(pred_pos, dtype=torch.float32)
pred_R = torch.tensor(pred_rot_matrix, dtype=torch.float32)

print(pred_t)
print(pred_R)

gt_t = torch.tensor(pose_data[nr][0]["cam_t_m2c"], dtype=torch.float32)
gt_R = torch.tensor(pose_data[nr][0]["cam_R_m2c"], dtype=torch.float32).reshape((3,3))

print(gt_t)
print(gt_R)


def load_model_points(ply_path, dtype=torch.float32):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    model_points_np = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    model_points = torch.tensor(model_points_np, dtype=dtype)
    return model_points

def get_ply_files():
        folder_path = "datasets/Linemod_preprocessed/models/"
        ply_objs = {}
        for i in range (1,16):
          file_name = "obj_"
          if i < 10:
            file_name = file_name + "0"
          file_name = file_name + str(i) + ".ply"
          ply_objs[i] = load_model_points(folder_path + file_name)
        return ply_objs

def compute_ADD(model_points, R_gt, t_gt, R_pred, t_pred):
      pts_gt = torch.matmul(R_gt, model_points.T).T + t_gt.view(1, 3)
      pts_pred = torch.matmul(R_pred, model_points.T).T + t_pred.view(1, 3)

      distances = torch.norm(pts_gt - pts_pred, dim=1)
      return torch.mean(distances).item()

ply_objs = get_ply_files()

model_points = ply_objs[int(id)]

add = compute_ADD(model_points, gt_R, gt_t, pred_R, pred_t)

print(add)
