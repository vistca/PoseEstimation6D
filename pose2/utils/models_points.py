from plyfile import PlyData
import torch
import numpy as np

def load_model_points(ply_path, dtype=torch.float32):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    model_points_np = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    model_points = torch.tensor(model_points_np, dtype=dtype)
    return model_points

def get_ply_files():
        folder_path = "./datasets/Linemod_preprocessed/models/"
        ply_objs = {}
        for i in range (1,16):
          file_name = "obj_"
          if i < 10:
            file_name = file_name + "0"
          file_name = file_name + str(i) + ".ply"
          ply_objs[i] = load_model_points(folder_path + file_name)
        return ply_objs