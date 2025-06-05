from pose2.pose_dataset import LinemodDataset
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from plyfile import PlyData

from pose2.models.bb8_2 import BB8Model_2

# # --- BB8Model (PoseModel) Class Definition ---
# # This class is included here to make the code self-contained,
# # as it was defined in a previous turn and is used by the training function.
# class BB8Model(nn.Module):
#     def __init__(self, num_objects=15):
#         super().__init__()
#         # We use ResNet18 as the backbone instead of VGG for greater efficiency
#         self.backbone = resnet18(pretrained=True)
#         self.backbone.fc = nn.Identity()  # We remove the final fully connected layer

#         # Head for predicting 2D points (8 corners * 2 coordinates)
#         self.bbox_head = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 16)# 8 points * 2 coordinates
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         bbox_pred = self.bbox_head(features)
#         #symmetry_pred = self.symmetry_head(features)
#         return bbox_pred #, symmetry_pred

#     # Removed get_3d_bbox_points from BB8Model as it's now in LinemodDataset





# --- train_model function ---

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

def compute_ADD(model_points, R_gt, t_gt, R_pred, t_pred):
      pts_gt = torch.matmul(R_gt, model_points.T).T + t_gt.view(1, 3)
      pts_pred = torch.matmul(R_pred, model_points.T).T + t_pred.view(1, 3)

      distances = torch.norm(pts_gt - pts_pred, dim=1)
      return torch.mean(distances).item()

def rescale_pred(preds: torch.Tensor, bboxes: torch.Tensor, nr_datapoints):

    for i in range(nr_datapoints):
        pred = preds[i]
        pred_points = pred.squeeze(0)

        nr_points = pred_points.size()[0] // 2

        x_min = bboxes[i, 0]
        y_min = bboxes[i, 1]
        x_max = bboxes[i, 2] 
        y_max = bboxes[i, 3]        
        
        width = x_max - x_min
        height = y_max - y_min

        for j in range(nr_points):
            pred_points[2*j] = pred_points[2*j] * width + x_min
            pred_points[2*j+1] = pred_points[2*j+1] * height + y_min

        preds[i] = pred_points

    return preds


def reconstruct_3d_points_from_pred(preds: torch.Tensor, models_points_3d, nr_datapoints, flatten=False):

    camera_matrix = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    result = torch.empty(nr_datapoints, 12, dtype=torch.float32)

    for i in range(nr_datapoints):
        pred = preds[i]
        pred_points = pred.squeeze(0)

        model_points_2d = np.empty((8,2))
        
        nr_points = pred_points.size()[0] // 2


        for j in range(nr_points):
            point_x = pred_points[2*j].item() 
            point_y = pred_points[2*j+1].item() 
            model_points_2d[j] = point_x, point_y

        model_points_3d = models_points_3d[i]

        success, pred_rvec, pred_pos = cv2.solvePnP(
            objectPoints=model_points_3d.numpy(),
            imagePoints=model_points_2d,
            cameraMatrix=camera_matrix,
            distCoeffs=np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        pred_rot_matrix, _ = cv2.Rodrigues(pred_rvec)
        pred_t = torch.tensor(pred_pos, dtype=torch.float32).squeeze()
        pred_R = torch.tensor(pred_rot_matrix, dtype=torch.float32).flatten()

        result[i] = torch.cat((pred_t, pred_R), dim=0)

    return result


def train_model(batch_size = 128, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration
    dataset_root = "./datasets/Linemod_preprocessed/" # Ensure this path is correct relative to your Colab environment
    learning_rate = 0.001
    checkpoint_dir = "./pose2/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    train_dataset = LinemodDataset(dataset_root, split='train')
    val_dataset = LinemodDataset(dataset_root, split='test')
    ply_objs = get_ply_files()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)
    print("Data loaders completed")

    # Model and optimizer
    model = BB8Model_2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error Loss for keypoint regression

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        train_loss = 0.0

        # Training loop with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images = batch['rgb'].to(device)
            targets = batch['points_2d'].to(device)

            pred_points = model(images) # Forward pass: predict 2D points and ignore symmetry output

            # Removed the redundant loop for selected_preds
            loss = criterion(pred_points, targets) # Calculate loss between predicted and target points
            optimizer.zero_grad() # Zero the gradients before backpropagation
            loss.backward() # Perform backpropagation
            optimizer.step() # Update model parameters
            train_loss += loss.item() # Accumulate training loss


        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        add_total = [0,0]
        add_objects = {}

        with torch.no_grad(): # Disable gradient calculation for validation
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                images = batch['rgb'].to(device)
                targets = batch['points_2d'].to(device)

                pred_points = model(images) # Forward pass

                # Removed the redundant loop for selected_preds
                loss = criterion(pred_points, targets) # Calculate validation loss
                val_loss += loss.item() # Accumulate validation loss

                bboxes = batch["bbox"]
                nr_datapoints = bboxes.shape[0]
                pred_points = rescale_pred(pred_points, bboxes, nr_datapoints)
                
                ids = batch["obj_id"]

                models_points_3d = batch["points_3d"] 
                gts_t = batch["translation"]
                gts_R = batch["rotation"]

                reconstruction_3d = reconstruct_3d_points_from_pred(pred_points, models_points_3d, nr_datapoints)

                for i in range(nr_datapoints):

                    pred_t = reconstruction_3d[i, :3]
                    pred_R = reconstruction_3d[i, 3:].reshape((3,3))

                    gt_t = gts_t[i]
                    gt_R = gts_R[i].reshape((3,3))

                    model_points = ply_objs[int(ids[i].item()+1)]

                    add = compute_ADD(model_points, gt_R, gt_t, pred_R, pred_t)

                    add_obj = add_objects.get(str(int(ids[i].item()+1)))
                    if not add_obj:
                      new_count = 1
                      new_val = add
                    else:
                      new_count = add_obj[0] + 1
                      new_val = add_obj[1] + add
                    add_objects[str(int(ids[i].item()+1))] = [new_count, new_val]
                    add_total = [add_total[0] + 1, add_total[1] + add]

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        avg_add_total = add_total[1] / add_total[0]
        for k,v in add_objects.items():
            avg_add_obj = v[1] / v[0]
            print(f"Obj: {k}, Avg ADD: {avg_add_obj}")
        print(f"Total average ADD: {avg_add_total}")

        # ⭐ Saving "best" model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            model_save_name = "bb8_mod_1"
            
            if os.path.exists(model_save_name):
                os.remove(model_save_name)
            save_path = f"./pose2/checkpoints/{model_save_name}.pt"
            torch.save(model.state_dict(), save_path)
            
            print("✅ New best model saved.")

    return model

# --- Main execution block (for demonstration) ---
# Set device


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# data_dir = "./dataset/Linemod_preprocessed/" # Make sure this path is correct for your dataset

if __name__ == "__main__":

    batch_size = 4
    num_epochs = 3

    # Step 1: Train the model
    print(f"Starting model training...")
    trained_model = train_model(batch_size=batch_size, num_epochs=batch_size)
    print(f"Training complete.")


