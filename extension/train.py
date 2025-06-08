import torch
from tqdm import tqdm


class Trainer():

    def __init__(self, model, optimizer, args, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.args = args
        self.scheduler = scheduler


    def train_one_epoch(self, train_loader, device):

        total_loss = 0.0
        nr_batches = 0
        self.model.train()

        progress_bar = tqdm(train_loader, desc=f"Training", ncols=100)

        for batch_id, batch in enumerate(progress_bar):

            inputs = {}
            inputs["rgb"] = batch["rgb"].to(device)
            inputs["depth"] = batch["depth"].to(device)
            inputs["bbox"] = batch["bbox"].to(device)
            inputs["obj_id"] = batch["obj_id"].to(device).long()

            pred_points = self.model(inputs) # Forward pass: predict 2D points and ignore symmetry output

            targets = batch['points_2d'].to(device)

            # Removed the redundant loop for selected_preds
            loss = self.loss_fn(pred_points, targets) # Calculate loss between predicted and target points
            self.optimizer.zero_grad() # Zero the gradients before backpropagation
            loss.backward() # Perform backpropagation
            self.optimizer.step() # Update model parameters
            total_loss += loss.item() # Accumulate training loss

            nr_batches += 1

            progress_bar.set_postfix(total=total_loss/nr_batches)


        avg_loss = total_loss / len(train_loader)

        self.scheduler.step()
        print(f"Lr is {self.scheduler._last_lr[0]}")
  
        return {
            "Training total_loss" : avg_loss,
        }
    
