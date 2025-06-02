from tqdm import tqdm
import time
import statistics
import torch
from utils.custom_loss import CustomLossFunctions
import numpy as np

class Trainer():

    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        #self.custom_loss_fn = CustomLossFunctions()
        self.scheduler = scheduler
        print("Trainer 2 in use!")


    def train_one_epoch(self, train_loader, device):
        
        self.model.train() # Maybe unnecessary?
        total_loss = 0.0
        nr_batches = 0
        losses_origin = np.array([0.0, 0.0, 0.0])

        timings = {"DL update iter" : [], "load" : [], "fit/loss" : [], "backprop" : []}

        progress_bar = tqdm(train_loader, desc="Training", ncols=100)
        start = time.perf_counter()

        for batch_id, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()#set_to_none=True)
            end = time.perf_counter()
            timings["DL update iter"].append(end - start)
            
            start = time.perf_counter()
            nr_datapoints = batch["rgb"].shape[0]
            targets = torch.empty(nr_datapoints, 12, device=device)
            inputs = []
            ids = []

            # We must be able to improve/remove this loop
            for i in range(nr_datapoints):
                translation = batch["translation"][i].to(device).unsqueeze(0) # Add batch dimension
                rotation = batch["rotation"][i].to(device).flatten().unsqueeze(0) # Add batch dimension    
                target = torch.cat((translation, rotation), dim=1)
                targets[i] = target            
            
            for i in range(nr_datapoints):
                input = {}
                #input["rgb"] = batch["rgb"][i].to(device).unsqueeze(0) # Add batch dimension
                #input["bbox"] = batch["bbox"][i].to(device).unsqueeze(0)  # Add batch dimension
                #input["obj_id"] = batch["obj_id"][i].to(device).long().unsqueeze(0)  # Add batch dimension
                input["t"] = batch["translation"][i].to(device).unsqueeze(0) # Add batch dimension
                input["R"] = batch["rotation"][i].to(device).flatten().unsqueeze(0) # Add batch dimension
                inputs.append(input)
                ids.append(str(int(batch["obj_id"][i].item()))) # stores the id as a string, this is later used for the custom loss function

            end = time.perf_counter()
            timings["load"].append(end - start)

            start = time.perf_counter()
            
            preds = self.model(inputs)
            #loss = self.custom_loss_fn.loss(preds, targets, ids, device)
            loss = self.loss_fn(preds, targets)

            #losses_origin += self.custom_loss_fn.get_losses()

            end = time.perf_counter()
            timings["fit/loss"].append(end - start)

            start = time.perf_counter()

            loss.backward()

            self.optimizer.step()
  

                #self.wandb_instance.log_metric({"Learning rate after batch" : self.scheduler._last_lr})


            end = time.perf_counter()
            timings["backprop"].append(end - start)

            # Track loss
            total_loss += loss.item()

            nr_batches += 1

            progress_bar.set_postfix(total=total_loss/nr_batches)


        avg_loss = total_loss / len(train_loader)
  
        return {
            "Training total_loss" : avg_loss,
            "DL update iter" : statistics.median(timings["DL update iter"]),
            "Time load_data" : statistics.median(timings["load"]),
            "Time fit/calc_loss" : statistics.median(timings["fit/loss"]),
            "Time backprop" : statistics.median(timings["backprop"])#,
            #"rot" : losses_origin[0] / nr_batches,
            #"pos" : losses_origin[1] / nr_batches,
            #"pen" : losses_origin[2] / nr_batches
        }
    
