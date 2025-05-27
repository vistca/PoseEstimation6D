from tqdm import tqdm
import time
import statistics
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Tester():

    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()


    def validate(self, dataloader, device, type):

        self.model.eval()
        val_loss = 0.0

        print(f"Starting {type}...")
        progress_bar = tqdm(dataloader, desc=type, ncols=100)

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

                self.model.train()
                preds = self.model(inputs)
                self.model.eval()

                #print(type(loss_dict), loss_dict)  # Debugging output
                loss = self.loss_fn(preds, targets)

                val_loss += loss

                progress_bar.set_postfix(total=val_loss/(batch_id + 1))

        avg_loss = val_loss / len(dataloader)
        
        return {
                f"{type} total_loss" : avg_loss
            }, avg_loss
    
