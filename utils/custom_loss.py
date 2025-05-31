import json
import numpy as np
import torch

class CustomLossFunctions():

    def __init__(self):
        with open("./datasets/Linemod_preprocessed/models/models_info.json", 'r') as f:
            models = json.load(f) 

        self.models = {}
        self.model_points = {}

        for id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]:
            model = models[id]
            self.models[id] = model

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

            corner_points = np.array([
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ], dtype=np.float32)

            self.model_points[id] = torch.tensor(corner_points)


    def loss(self, preds, targets, ids, device):

        batch_size = torch.tensor(preds.shape[0]).to(device)
        total_loss = torch.tensor(0.0, dtype=float).to(device) # 0.0

        for i in range(batch_size):
            t_pred = preds[i, :3].unsqueeze(1).to(device)
            R_pred = preds[i, 3:].reshape(3, 3).to(device)

            t_gt = targets[i, :3].unsqueeze(1).to(device)
            R_gt = targets[i, 3:].reshape(3, 3).to(device)

            model_id = ids[i]
            pts = self.model_points[model_id].T.to(device)

            pred_pts = (R_pred @ pts + t_pred).to(device)
            gt_pts = (R_gt @ pts + t_gt).to(device)

            dist = torch.norm(pred_pts - gt_pts, dim=0).mean().to(device)
            total_loss += dist

        return total_loss / batch_size

        
# loss_function = CustomLossFunctions()

# preds = np.array([
#                 [0,0,0, 1,0,0, 0,1,0, 0,0,1],
#                 #[1,1,1, 1,0,0, 0,1,0, 0,0,1],
#             ], dtype=np.float32)
# preds = torch.tensor(preds)

# targets = np.array([
#                 [10,10,0, 1,0,0, 0,1,0, 0,0,1],
#                 #[1,1,1, 1,0,0, 0,1,0, 0,0,1],
#             ], dtype=np.float32)
# targets = torch.tensor(targets)

# ids = ["3"]

# print(loss_function.loss(preds, targets, ids))

