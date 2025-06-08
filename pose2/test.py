from pose2.utils.models_points import get_ply_files
from pose2.utils.data_reconstruction import reconstruct_3d_points_from_pred, rescale_pred
from pose2.utils.add_calc import compute_ADD
import torch
from tqdm import tqdm


class Tester():

    def __init__(self, model, args):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.args = args
        self.ply_objs = get_ply_files()


    def validate(self, val_loader, device, type="Val"):
        # Validation phase
        self.model.eval() # Set model to evaluation mode        
        val_loss = 0.0
        nr_batches = 0
        below_2cm = [0,0]
        add_total = [0,0]
        add_objects = {}
        dia_10_objects = {}
        below_dia = [0,0]

        with torch.no_grad(): # Disable gradient calculation for validation
            if type == "Val":
                desc = f" Val"
            else:
                desc = "Test"
            progress_bar = tqdm(val_loader, desc=desc, ncols=100)


            for batch_id, batch in enumerate(progress_bar):

                inputs = {}
                inputs["rgb"] = batch["rgb"].to(device)
                #inputs["bbox"] = batch["bbox"].to(device)
                #inputs["obj_id"] = batch["obj_id"].to(device).long()

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
                diameters = batch["diameter"]

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

                    below_2cm[1] = below_2cm[1] + 1
                    if add < 20:
                        below_2cm[0] = below_2cm[0] + 1

                    dia_10_object = dia_10_objects.get(str(int(ids[i].item()+1)))
                    diameter = diameters[i]
                    new_low = 1 if add < 0.1*diameter else 0
                    if not dia_10_object:
                        new_count = 1
                        new_val = new_low
                    else:
                        new_count = dia_10_object[0] + 1
                        new_val = dia_10_object[1] + new_low
                    dia_10_objects[str(int(ids[i].item()+1))] = [new_count, new_val]
                    below_dia = [below_dia[0] + new_low, below_dia[1] + 1]


        avg_val_loss = val_loss / len(val_loader)
        avg_add_total = add_total[1] / add_total[0]
        percentage_below_2cm = 100*below_2cm[0]/below_2cm[1]
        percentage_below_10_dia = 100*below_dia[0]/below_dia[1]
        
        for k,v in add_objects.items():
            avg_add_obj = v[1] / v[0]
            print(f"Obj: {k}, Avg ADD: {avg_add_obj}, num obj: {v[0]}")
        print(f"Total average ADD: {avg_add_total}")

        print("-"*25)
        
        for k,v in dia_10_objects.items():
            perc_below_dia = v[1] / v[0]
            print(f"Obj: {k}, Percentage below 10% of diameter: {perc_below_dia}, num obj: {v[0]}")
        print(f"Total percentage below 10% of diameter: {percentage_below_10_dia}")
        print(f"Percentage below 2cm : {percentage_below_2cm}")

        return {
                f"{type} total_loss" : avg_val_loss,
                f"{type} total_ADD" : avg_add_total,
                f"{type} below_2cm" : percentage_below_2cm,
                f"{type} below_10%_diameter" : percentage_below_10_dia
            }, avg_add_total

