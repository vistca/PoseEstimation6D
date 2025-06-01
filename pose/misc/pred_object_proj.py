from PIL import Image, ImageDraw
import json
from pose.models.efficientNet import CustomEfficientNet
from pose.models.resnet import CustomResNet50
import torchvision.transforms as transforms
import torch
import numpy as np
from utils.projections import project_to_2d

# To run from command line:
# python -m pose.misc.pred_object_proj


# ----------------------------- functions -----------------------------

def draw_bcube(draw, pos, rot, color):
    coord = np.array(pos).T
    rotation_matrix = np.array(rot).reshape(3, 3)

    model = models[obj_id]
    edges = model_to_edges(model)

    rotate_edges(edges, rotation_matrix)
    transpose_edges(edges, coord)
    proj_edges = project_edges(edges)

    for edge in proj_edges:
        draw.line(edge, fill=color, width=1)


def model_to_edges(model):
    edges = []

    x_min = model["min_x"]
    x_size = model["size_x"]
    x_max = x_min + x_size

    y_min = model["min_y"]
    y_size = model["size_y"]
    y_max = y_min + y_size

    z_min = model["min_z"]
    z_size = model["size_z"]
    z_max = z_min + z_size

    # The coordinates of the edges of the bounding cube
    # l stats for lower and h higher value for x, y and z
    lll = np.array([x_min, y_min, z_min]).T
    llh = np.array([x_min, y_min, z_max]).T
    lhl = np.array([x_min, y_max, z_min]).T
    lhh = np.array([x_min, y_max, z_max]).T
    hll = np.array([x_max, y_min, z_min]).T
    hlh = np.array([x_max, y_min, z_max]).T
    hhl = np.array([x_max, y_max, z_min]).T
    hhh = np.array([x_max, y_max, z_max]).T

    # The edges of the cube
    edges.append([lll, lhl])
    edges.append([lll, hll])
    edges.append([hhl, lhl])
    edges.append([hhl, hll])

    edges.append([lll, llh])
    edges.append([lhl, lhh])
    edges.append([hhl, hhh])
    edges.append([hll, hlh])

    edges.append([llh, lhh])
    edges.append([llh, hlh])
    edges.append([hhh, lhh])
    edges.append([hhh, hlh])

    return edges


def project_edges(edges):
    proj_edges = []
    for edge in edges:
        point1 = project_to_2d(edge[0])
        point2 = project_to_2d(edge[1])
        proj_edges.append([point1, point2])
    return proj_edges


def transpose_edges(edges, increment_coord):
    for i in range(len(edges)):
        for j in range(2):
            edges[i][j] += increment_coord


def rotate_edges(edges, rotation_matrix):
    for i in range(len(edges)):
        for j in range(2):
            edges[i][j] = rotation_matrix @ edges[i][j]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pose_model_3"

obj_id = "2"
dir = "0" * (2 - len(obj_id)) + obj_id
nr = "1000"
img_nr = "0" * (4 - len(nr))
img_nr = img_nr + nr

img_path = "./datasets/Linemod_preprocessed/data/" + dir + "/rgb/"
info_path = "./datasets/Linemod_preprocessed/data/" + dir + "/"
model_file_path = "./datasets/Linemod_preprocessed/models/models_info.json"
image_name = img_nr + ".png"
json_name = "gt.json"


img = Image.open(img_path + image_name).convert("RGB")

pose_data = {}

with open(info_path + json_name, 'r') as f:
    pose_data = json.load(f)

with open(model_file_path, 'r') as f:
    models = json.load(f) 

# [x_left, y_top, x_width, y_height]
b = pose_data[nr][0]["obj_bb"] # bounding box
id = pose_data[nr][0]["obj_id"]

crop = img.crop((b[0], b[1], b[0]+b[2], b[1]+b[3]))

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

resize_format = (300, 300)
crop = crop.resize(resize_format)

#crop.show()


data_point = {}
data_point["rgb"] = tensor_transform(crop).unsqueeze(0)
data_point["bbox"] = torch.tensor(b, dtype=torch.float).unsqueeze(0)
data_point["obj_id"] = torch.tensor(id, dtype=torch.long).unsqueeze(0)


model = CustomResNet50()
model.load_state_dict(torch.load(f"./checkpoints/{model_name}.pt", map_location=device))
model.eval()

pred = model([data_point]).squeeze().tolist()


pos_pred = pred[:3]
rot_pred = pred[3:]

pos_gt = pose_data[nr][0]["cam_t_m2c"] # 3d coordinates
rot_gt = pose_data[nr][0]["cam_R_m2c"] # rotation matrix

draw = ImageDraw.Draw(img)

color1 = "#FF1AF4"
color2 = "#FF9900"

print(pos_pred)
print(pos_gt)

print("-"*15)

print(rot_pred)
print(rot_gt)

draw_bcube(draw, pos_pred, rot_pred, color1)
draw_bcube(draw, pos_gt, rot_gt, color2)

img.show()




