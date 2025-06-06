from pose2.utils.models_points import get_ply_files
from pose2.utils.data_reconstruction import reconstruct_3d_points_from_pred, rescale_pred
from pose2.utils.add_calc import compute_ADD
import torch
from tqdm import tqdm

class Tester():

    def __init__(self, model, epochs):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs = epochs
        self.ply_objs = get_ply_files()

    def validate(self, val_loader, device, type="Val"):
        # Validation phase
        self.model.eval() # Set model to evaluation mode        
        val_loss = 0.0
        nr_batches = 0
        add_total = [0,0]
        add_objects = {}

        with torch.no_grad(): # Disable gradient calculation for validation
            if type == "Val":
                desc = f"Val"
            else:
                desc = "Test"
            progress_bar = tqdm(val_loader, desc=desc, ncols=100)

            for batch_id, batch in enumerate(progress_bar):

                inputs = {}
                inputs["rgb"] = batch["rgb"].to(device)
                inputs["depth"] = batch["depth"].to(device)
                inputs["bbox"] = batch["bbox"].to(device)
                inputs["obj_id"] = batch["obj_id"].to(device).long()

                pred_points = self.model(inputs) # Forward pass

                targets = batch['points_2d'].to(device)

                # Removed the redundant loop for selected_preds
                loss = self.loss_fn(pred_points, targets) # Calculate validation loss
                val_loss += loss.item() # Accumulate validation loss
                nr_batches += 1

                progress_bar.set_postfix(total=val_loss/nr_batches)

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

                    model_points = self.ply_objs[int(ids[i].item()+1)]

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


        avg_val_loss = val_loss / len(val_loader)
        avg_add_total = add_total[1] / add_total[0]
        
        for k,v in add_objects.items():
            avg_add_obj = v[1] / v[0]
            print(f"Obj: {k}, Avg ADD: {avg_add_obj}, num obj: {v[0]}")
        print(f"Total average ADD: {avg_add_total}")

        return {
                f"{type} total_loss" : avg_val_loss,
                f"{type} total_ADD" : avg_add_total
            }, avg_add_total