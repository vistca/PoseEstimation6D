from tqdm import tqdm
import time
import statistics
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from plyfile import PlyData

class Tester():

    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def compute_ADD(self, model_points, R_gt, t_gt, R_pred, t_pred):
      pts_gt = torch.matmul(R_gt, model_points.T).T + t_gt.view(1, 3)
      pts_pred = torch.matmul(R_pred, model_points.T).T + t_pred.view(1, 3)

      distances = torch.norm(pts_gt - pts_pred, dim=1)
      return torch.mean(distances).item()

    def load_model_points(self, ply_path, dtype=torch.float32):
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex']
        model_points_np = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
        model_points = torch.tensor(model_points_np, dtype=dtype)
        return model_points
    
    def get_ply_file(self, obj_id):
        if (obj_id < 10):
            return "obj_0" + str(obj_id) + ".ply"
        return "obj_" + str(obj_id) + ".ply"

    def get_ply_files(self):
        folder_path = "datasets/Linemod_preprocessed/models/"
        ply_objs = {}
        for i in range (1,16):
          file_name = "obj_"
          if i < 10:
            file_name = file_name + "0"
          file_name = file_name + str(i) + ".ply"
          ply_objs[i] = self.load_model_points(folder_path + file_name)
        return ply_objs

    def validate(self, dataloader, device, type):

        self.model.eval()
        val_loss = 0.0

        print(f"Starting {type}...")
        progress_bar = tqdm(dataloader, desc=type, ncols=100)
        add_total = [0,0]
        add_objects = {}

        ply_objs = self.get_ply_files()

        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):

                nr_datapoints = batch["rgb"].shape[0]
                targets = torch.empty(nr_datapoints, 12, device=device)
                inputs = []

                # We must be able to improve/remove this loop                
                for i in range(nr_datapoints):
                    translation = batch["translation"][i].to(device).unsqueeze(0) # Add batch dimension
                    rotation = batch["rotation"][i].to(device).flatten().unsqueeze(0) # Add batch dimension    
                    target = torch.cat((translation, rotation), dim=1)
                    targets[i] = target            
                
                for i in range(nr_datapoints):
                    input = {}
                    input["rgb"] = batch["rgb"][i].to(device).unsqueeze(0) # Add batch dimension
                    input["bbox"] = batch["bbox"][i].to(device).unsqueeze(0)  # Add batch dimension
                    input["obj_id"] = batch["obj_id"][i].to(device).long().unsqueeze(0)  # Add batch dimension
                    inputs.append(input)

                # Forward pass

                # weird quirk with eval() only returning predictions. Probably
                # bad practice to elvaluate in train() mode.

                # doing it like this takes forever, might need to check this and update it accordingly

                preds = self.model(inputs)

                #print(type(loss_dict), loss_dict)  # Debugging output
                loss = self.loss_fn(preds, targets)

                # Calculate the ADD metric
                models_folder = "./datasets/Linemod_preprocessed/models/"
                for i in range(nr_datapoints):
                    pred = preds[i]
                    gt = targets[i]
                    t_pred = pred[:3].reshape(3,1).to(device)
                    R_pred = pred[3:].reshape(3,3).to(device)
                    t_gt = gt[:3].reshape(3,1).to(device)
                    R_gt = gt[3:].reshape(3,3).to(device)
                    obj_id = int(batch["obj_id"][i].item())
                    ply_file = self.get_ply_file(obj_id)
                    model_points = ply_objs[obj_id].to(device)
                    add = self.compute_ADD(model_points, R_gt, t_gt, R_pred, t_pred)
                    #print("The add is: " + str(add))
                    add_obj = add_objects.get(str(obj_id))
                    if not add_obj:
                      new_count = 1
                      new_val = add
                    else:
                      new_count = add_obj[0] + 1
                      new_val = add_obj[1] + add
                    add_objects[str(obj_id)] = [new_count, new_val]
                    add_total = [add_total[0] + 1, add_total[1] + add]
                val_loss += loss

                progress_bar.set_postfix(total=val_loss/(batch_id + 1))
                
        avg_loss = val_loss / len(dataloader)
        avg_add_total = add_total[1] / add_total[0]
        for k,v in add_objects.items():
          avg_add_obj = v[1] / v[0]
          print(f"Obj: {k}, Avg ADD: {avg_add_obj}")
        print(f"Total average ADD: {avg_add_total}")
        
        result_dict = {f"{type} total_loss" : avg_loss}

        return result_dict
    
