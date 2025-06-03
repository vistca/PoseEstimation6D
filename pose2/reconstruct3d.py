import torch
import numpy as np
import cv2

def reconstruct_3d_points_from_pred(preds: torch.Tensor, models_points_3d, bboxes, nr_datapoints):

    camera_matrix = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    result = torch.empty((nr_datapoints, 12), dtype=np.float32)

    for i in range(nr_datapoints):
        pred = preds[i]
        pred_points = pred.squeeze(0)

        model_points_2d = np.empty((8,2))
        nr_points = pred_points.size()[0] // 2

        x_min, y_min, x_max, y_max = bboxes[i]        
        width = x_max - x_min
        height = y_max - y_min

        for i in range(nr_points):
            point_x = pred_points[2*i].item() * width + x_min
            point_y = pred_points[2*i+1].item() * height + y_min
            model_points_2d[i] = point_x, point_y

        model_points_3d = model_points_3d[i]

        success, pred_rvec, pred_pos = cv2.solvePnP(
            objectPoints=model_points_3d,
            imagePoints=model_points_2d,
            cameraMatrix=camera_matrix,
            distCoeffs=np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        pred_rot_matrix, _ = cv2.Rodrigues(pred_rvec)
        pred_t = torch.tensor(pred_pos, dtype=torch.float32)
        pred_R = torch.tensor(pred_rot_matrix, dtype=torch.float32)
        result[i] = torch.cat((pred_t, pred_R), dim=0)

    return result

