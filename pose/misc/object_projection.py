
import numpy as np
from PIL import Image, ImageDraw
import json


def project_to_2d(coord):
    K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ]) # The camera matrix

    # The two dimensional coordinates derived through projection
    x = (K[0, 0] * coord[0] / coord[2]) + K[0, 2]
    y = (K[1, 1] * coord[1] / coord[2]) + K[1, 2]

    return (x, y)


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


def rotat_edges(edges, rotation_matrix):
    for i in range(len(edges)):
        for j in range(2):
            edges[i][j] = rotation_matrix @ edges[i][j]



obj_id = "14"
dir = "0" * (2 - len(obj_id)) + obj_id

nr = "1170"
img_nr = "0" * (4 - len(nr)) + nr

img_path = "./datasets/Linemod_preprocessed/data/" + dir + "/rgb/"
info_path = "./datasets/Linemod_preprocessed/data/" + dir + "/"
model_file_path = "./datasets/Linemod_preprocessed/models/models_info.json"
image_name = img_nr + ".png"
json_name = "gt.json"


img = Image.open(img_path + image_name).convert("RGB")


with open(info_path + json_name, 'r') as f:
    pose_data = json.load(f)

with open(model_file_path, 'r') as f:
    models = json.load(f) 


coord = pose_data[nr][0]["cam_t_m2c"] # 3d coordinates
rotation = pose_data[nr][0]["cam_R_m2c"] # rotation matrix

coord = np.array(coord).T
rotation_matrix = np.array(rotation).reshape(3, 3)

model = models[obj_id]
edges = model_to_edges(model)

rotat_edges(edges, rotation_matrix)
transpose_edges(edges, coord)
proj_edges = project_edges(edges)


draw = ImageDraw.Draw(img)
for edge in proj_edges:
        draw.line(edge, fill="#FF1AF4", width=1)


img.show()

