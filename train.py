from tqdm import tqdm

class Trainer():

    def __init__(self, model, optimizer, wandb_instance, epochs):
        self.model = model
        self.optimizer = optimizer
        self.wandb_instance = wandb_instance
        self.epochs = epochs

    def train(self, dataloader, device):
        
        train_losses = []

        for epoch in range(self.epochs):

            self.model.train()
            total_loss = 0.0
            nr_batches = 0

            for batch in tqdm(dataloader):
                images = batch["rgb"].to(device)
                targets = []

                # We must be able to improve/remove this loop
                for i in range(images.shape[0]):
                    target = {}
                    target["boxes"] = batch["bbox"][i].to(device).unsqueeze(0)  # Add batch dimension
                    target["labels"] = batch["obj_id"][i].to(device).long().unsqueeze(0)  # Add batch dimension
                    targets.append(target)
                

                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track loss
                total_loss += loss.item()

                nr_batches += 1
                break

            avg_loss = total_loss / len(dataloader)
            train_losses.append(avg_loss)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

            self.wandb_instance.log_metric({"training_loss" : avg_loss})
    
