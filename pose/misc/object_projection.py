
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

    return [x, y]


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





obj_id = "8"
dir = "0" * (2 - len(obj_id)) + obj_id

nr = "1156"
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
rotation = np.array(rotation).reshape(3, 3)



model = models[obj_id]
print(model)

print(model_to_edges(model))



x, y = project_to_2d(coord)

#print(f"(x, y) = ({x}, {y})")

radius = 2 
dot_color = "red"

x0 = x - radius
y0 = y - radius
x1 = x + radius
y1 = y + radius

draw = ImageDraw.Draw(img)
draw.ellipse((x0, y0, x1, y1), fill=dot_color)

#print(f"[x, y, z] = {coord}")


#img.show()

