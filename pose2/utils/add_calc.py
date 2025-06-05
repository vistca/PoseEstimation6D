import torch

def compute_ADD(model_points, R_gt, t_gt, R_pred, t_pred):
      pts_gt = torch.matmul(R_gt, model_points.T).T + t_gt.view(1, 3)
      pts_pred = torch.matmul(R_pred, model_points.T).T + t_pred.view(1, 3)

      distances = torch.norm(pts_gt - pts_pred, dim=1)
      return torch.mean(distances).item()
