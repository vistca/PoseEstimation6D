from tqdm import tqdm
import time
import statistics
import torch
from torch.amp import autocast, GradScaler

class Trainer():

    def __init__(self, model, optimizer, epochs):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss_fn = torch.nn.MSELoss()
        self.scaler = GradScaler()



    def train_one_epoch(self, train_loader, device):

        self.model.train() # Maybe unnecessary?
        total_loss = 0.0
        nr_batches = 0

        timings = {"DL update iter" : [], "load" : [], "fit/loss" : [], "backprop" : []}

        progress_bar = tqdm(train_loader, desc="Training", ncols=100)
        start = time.perf_counter()

        for batch_idx, batch in enumerate(progress_bar):
            end = time.perf_counter()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            timings["DL update iter"].append(end - start)
            
            start = time.perf_counter()
            nr_datapoints = batch["rgb"].shape[0]
            targets = torch.empty(nr_datapoints, 12, device=device)
            inputs = []

            for i in range(nr_datapoints):
                translation = batch["translation"][i].to(device).unsqueeze(0) # Add batch dimension
                rotation = batch["rotation"][i].to(device).flatten().unsqueeze(0) # Add batch dimension    
                target = torch.cat((translation, rotation), dim=1)
                targets[i] = target            
            

            # retrieve both info
            # pass through each cnn
            # use torch.concat to fuse outputs
                # maybe extra global features if we want
            # send the concat to the posemodel
            # calculate the loss for the posemodel

            inputs = {"rgb" : [], "depth" : [], "bbox": [], "obj_id" : []}
            for i in range(nr_datapoints):
                inputs["rgb"] = inputs.get("rgb").append(batch["rgb"][i].to(device).unsqueeze(0))
                inputs["depth"] = inputs.get("depth").append(batch["depth"][i].to(device).unsqueeze(0))
                inputs["bbox"] = inputs.get("bbox").append(batch["bbox"][i].to(device).unsqueeze(0))
                inputs["obj_id"] = inputs.get("obj_id").append(batch["obj_id"][i].to(device).unsqueeze(0))

            
            end = time.perf_counter()
            timings["load"].append(end - start)

            start = time.perf_counter()
            # TODO: Why cuda, change call to self.model to instead calculate custom mse loss
            # Using mixed precision training
            if device.type == 'cuda':
                with autocast(device.type):
                    preds = self.model.forward()
                    loss = self.loss_fn(preds, targets)
            else:
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
            
            

            end = time.perf_counter()
            timings["fit/loss"].append(end - start)

            start = time.perf_counter()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            end = time.perf_counter()
            timings["backprop"].append(end - start)

            # Track loss
            total_loss += loss.item()

            nr_batches += 1

            progress_bar.set_postfix(total=total_loss/nr_batches)

        avg_loss = total_loss / len(train_loader)

        result_dict = {}
        result_dict["Training total_loss"] = avg_loss
        result_dict["DL update iter"] = statistics.median(timings["DL update iter"]),
        result_dict["Time load_data"] = statistics.median(timings["load"]),
        result_dict["Time fit/calc_loss"] = statistics.median(timings["fit/loss"]),
        result_dict["Time backprop"] = statistics.median(timings["backprop"]),


        return result_dict #avg_loss
    
