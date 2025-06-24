import numpy as np
import torch

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


def direction_from_2d(coord):
    K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ])

    x_dir = (coord[0] + K[0, 2]) / K[0, 0]
    y_dir = (coord[0] + K[1, 2]) / K[1, 1]

    return (x_dir, y_dir)


def get_3d_point(coord_2d, depth):
    x_and_y = direction_from_2d(coord_2d) * depth
    return (x_and_y[0], x_and_y[1], depth)


def direction_from_bbox(bbox):
    # bbox = [x_left, y_top, x_width, y_height]
    x_left = bbox[0]
    y_top = bbox[1]
    x_right = x_left + bbox[2]
    y_bottom = y_top + bbox[3]
    
    directions = np.empty((8, 1))

    coords = [(x_left, y_top), (x_left, y_bottom), (x_right, y_top), (x_right, y_bottom)]

    for i, coord in enumerate(coords):
        directions[2*i : 2*i + 2] = torch.tensor(direction_from_2d(coord)).unsqueeze(1)

    return directions

def directions_from_bboxs(bboxes: torch.Tensor):
    nr_bboxes = bboxes.shape[0]

    directions = np.empty((nr_bboxes,8))

    for i in range(nr_bboxes):
        directions[i] = direction_from_bbox(bboxes[i]).squeeze()

    return torch.tensor(directions)


