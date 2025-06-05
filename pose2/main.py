from utils.runtime_args import add_runtime_args
from pose2.pose_dataset import LinemodDataset
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml


from utils.optimizer_loader import OptimLoader
from pose2.utils.models_points import get_ply_files
from pose2.utils.data_reconstruction import reconstruct_3d_points_from_pred, rescale_pred
from pose2.utils.add_calc import compute_ADD
from pose2.models.model_creator import create_model



def run_program(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration
    dataset_root = "./datasets/Linemod_preprocessed/" # Ensure this path is correct relative to your Colab environment
    batch_size = args.bs
    checkpoint_dir = "./pose2/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    with open('./config/global_runtime_config.yaml') as f:
            config_dict = yaml.safe_load(f)

    # Setup the model
    selected_model = create_model(args.mod)
    model = selected_model.to(device)

    split_percentage = {
                        "train_%" : config_dict["train_%"],
                        "test_%" : config_dict["test_%"],
                        "val_%" : config_dict["val_%"],
                        }
    
    train_dataset = LinemodDataset(dataset_root, split_percentage, model.get_dimension(), split="train")
    test_dataset = LinemodDataset(dataset_root, split_percentage, model.get_dimension(), split="test")
    val_dataset = LinemodDataset(dataset_root, split_percentage, model.get_dimension(), split="val")

    #train_dataset = LinemodDataset(dataset_root, split='train')
    #val_dataset = LinemodDataset(dataset_root, split='test')
    ply_objs = get_ply_files()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Data loaders completed")


    # optimizer and loss function
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimloader = OptimLoader(args.optimizer, model_params, args.lr)
    optimizer = optimloader.get_optimizer()
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        train_loss = 0.0

        # Training loop with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train"):

            #images = batch['rgb'].to(device)
            inputs = {}
            inputs["rgb"] = batch["rgb"].to(device)
            inputs["bbox"] = batch["bbox"].to(device)
            inputs["obj_id"] = batch["obj_id"].to(device).long()

            pred_points = model(inputs) # Forward pass: predict 2D points and ignore symmetry output

            targets = batch['points_2d'].to(device)

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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val"):
                
                #images = batch['rgb'].to(device)
                inputs = {}
                inputs["rgb"] = batch["rgb"].to(device)
                inputs["bbox"] = batch["bbox"].to(device)
                inputs["obj_id"] = batch["obj_id"].to(device).long()

                pred_points = model(inputs) # Forward pass

                targets = batch['points_2d'].to(device)

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
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        avg_add_total = add_total[1] / add_total[0]
        for k,v in add_objects.items():
            avg_add_obj = v[1] / v[0]
            print(f"Obj: {k}, Avg ADD: {avg_add_obj}")
        print(f"Total average ADD: {avg_add_total}")

        # ⭐ Saving "best" model
        if avg_val_loss < best_val_loss and args.sm != "":
            best_val_loss = avg_val_loss
            
            if os.path.exists(args.sm):
                os.remove(args.sm)
            save_path = f"./pose2/checkpoints/{args.sm}.pt"
            torch.save(model.state_dict(), save_path)
            
            print("✅ New best model saved.")


if __name__ == "__main__":

    args = add_runtime_args()
    run_program(args)

# python -m pose2.main --lr 0.001 --bs 16 --epochs 2 --mod res18

