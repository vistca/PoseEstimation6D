from tqdm import tqdm
import time
import statistics
import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer():

    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.scaler = GradScaler()
        self.scheduler = scheduler



    def train_one_epoch(self, train_loader, device):

        self.model.train() # Maybe unnecessary?
        total_loss = 0.0
        nr_batches = 0

        timings = {"DL update iter" : [], "load" : [], "fit/loss" : [], "backprop" : []}

        progress_bar = tqdm(train_loader, desc="Training", ncols=100)
        start = time.perf_counter()

        for batch_id, batch in enumerate(progress_bar):
            self.optimizer.zero_grad(set_to_none=True)
            end = time.perf_counter()
            timings["DL update iter"].append(end - start)
            
            start = time.perf_counter()
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
            
            end = time.perf_counter()
            timings["load"].append(end - start)

            start = time.perf_counter()
            # TODO: Why cuda, change call to self.model to instead calculate custom mse loss
            # Using mixed precision training
            if device.type == 'cuda':
                with autocast(device.type):
                    preds = self.model(inputs)  
                    loss = self.loss_fn(preds, targets)
            else:
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
            

            self.model.eval()

            

            end = time.perf_counter()
            timings["fit/loss"].append(end - start)

            start = time.perf_counter()

            self.scaler.scale(loss).backward()
            scaled_factor = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if not scaled_factor <= self.scaler.get_scale() and self.scheduler:
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
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
            "Time backprop" : statistics.median(timings["backprop"])
        }
    
