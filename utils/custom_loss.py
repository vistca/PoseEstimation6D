import json
import numpy as np
import torch

class CustomLossFunctions():

    def __init__(self):
        with open("./datasets/Linemod_preprocessed/models/models_info.json", 'r') as f:
            models = json.load(f) 

        self.batch_size = 1

        self.models = {}
        self.model_points = {}

        self.losses = [0, 0, 0]
        self.total_loss = 0

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

    def log_losses(self, rot, pos, pen):
        self.losses[0] += rot.item()
        self.losses[1] += pos.item()
        self.losses[2] += pen.item()
        self.total_loss += rot.item() + pos.item() + pen.item()

    def reset_log(self):
        self.losses = np.array([0.0, 0.0, 0.0])
        self.total_loss = 0.0

    def get_losses(self):
        return self.losses / (self.batch_size * self.total_loss)

    def loss_old(self, preds, targets, ids, device):
        
        self.batch_size = torch.tensor(preds.shape[0]).to(device)
        total_loss = torch.tensor(0.0, dtype=float).to(device)

        for i in range(self.batch_size):
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
  
        return total_loss / self.batch_size
    

    def rot_matrix_prop_penalty(self, R):
        # R is a 3x3 matrix (torch.Tensor)
        RtR = torch.matmul(R.T, R)
        identity = torch.eye(3, device=R.device)
        diff = (RtR - identity).flatten()
        return torch.norm(diff)
    
    
    def loss(self, preds, targets, ids, device="cpu"): # the default is set to cpu for testing, in training it should always be provided

        self.reset_log()
        batch_size = torch.tensor(preds.shape[0]).to(device)
        total_loss = torch.tensor(0.0, dtype=float).to(device)

        for i in range(batch_size):
            t_pred = preds[i, :3].unsqueeze(1).to(device)
            R_pred = preds[i, 3:].reshape(3, 3).to(device)

            t_gt = targets[i, :3].unsqueeze(1).to(device)
            R_gt = targets[i, 3:].reshape(3, 3).to(device)

            model_id = ids[i]
            pts = self.model_points[model_id].T.to(device)

            pred_pts = (R_pred @ pts).to(device)
            gt_pts = (R_gt @ pts).to(device)

            rot_error = torch.norm(pred_pts - gt_pts, dim=0).mean().to(device)
            pos_error = torch.norm(t_pred - t_gt).to(device)
            
            ortho_penalty = self.rot_matrix_prop_penalty(R_pred)

            rot = 0.1 * rot_error**2
            pos = pos_error**2
            pen = 0.01 * ortho_penalty
            self.log_losses(rot, pos, pen)
            total_loss += rot + pos + pen

        return total_loss / batch_size

        
# loss_function = CustomLossFunctions()

# preds = np.array([
#                 [0,0,0, 1,0,0, 0,1,0, 0,0,1],
#                 [3,1,1, 1,0,0, 0,1,0, 0,0,1],
#             ], dtype=np.float32)
# preds = torch.tensor(preds)

# targets = np.array([
#                 [10,0,0, 1,0,0, 0,1,0, 0,0,1],
#                 [1,1,1, 1,0,0, 0,1,0, 0,0,1],
#             ], dtype=np.float32)
# targets = torch.tensor(targets)

# ids = ["3", "10"]

# print(loss_function.loss(preds, targets, ids).item())

