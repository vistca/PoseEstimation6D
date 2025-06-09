import numpy as np

def get_3d_bbox_points(models_info, obj_id):
        """Obtains the 3D bounding box points for an object from dataset's models_info"""
        # models_info is already loaded in __init__
        info = models_info[obj_id]
        x_size = info['size_x']
        y_size = info['size_y']
        z_size = info['size_z']

        half_x = x_size / 2
        half_y = y_size / 2
        half_z = z_size / 2

        points_3d = np.array([
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [half_x, half_y, half_z],
            [-half_x, half_y, half_z]
        ], dtype=np.float32)

        return points_3d