from train import Trainer
from test import Tester
import time
import datetime
import torch
import os

class TTH():
    def __init__(self, model, optimizer, wandb_instance, epochs):
        self.model = model
        self.optimizer = optimizer
        self.wandb = wandb_instance
        self.epochs = epochs
        self.trainer = Trainer(model, optimizer, wandb_instance, epochs)
        self.tester = Tester(model, wandb_instance)

    def save_model(self, path, prev_losses, curr_loss):
        if curr_loss < min(prev_losses):
            if os.path.exists(path):
                os.remove(path)
            torch.save(self.model.state_dict(), 'checkpoints/'+ path + ".pt")
            return "Model saved"
        return "Model not saved"


    def train_test_val_model(self, train_dl, val_dl, test_dl, device, save_path):
        start = time.perf_counter()

        losses = {"train_loss" : [], "val_loss" : [], "train_mAP" : [], "val_mAP" : []}

        for epoch in range(self.epochs):
            
            print(f"\nStarting epoch {epoch+1}/{self.epochs}")

            train_avg_loss, train_avg_mAP = self.trainer.train_one_epoch(train_dl, device)
            losses["train_loss"].append(train_avg_loss)
            losses["train_mAP"].append(train_avg_loss)

            val_avg_loss, val_avg_mAP = self.tester.validate(val_dl, device, 'Validation')
            losses["val_loss"].append(train_avg_loss)
            losses["val_mAP"].append(train_avg_loss)
            
            end = time.perf_counter()
            time_diff = end - start
            time_to_run = str(datetime.timedelta(seconds=time_diff))

            save_status = "Unapplicable"
            if save_path != "":
                save_status = self.save_model(save_path, 
                                              losses["val_loss"],
                                              val_avg_loss)

            print(f"Epoch statistics for epoch {epoch+1}/{self.epochs} \n" \
                   "---------------------------------- \n" \
                   f"Average training loss: {train_avg_loss:.4f} \n" \
                   f"Average training mAP: {train_avg_mAP:.4f} \n" \
                   f"Average validation loss: {val_avg_loss:.4f} \n" \
                   f"Average validation mAP: {val_avg_mAP:.4f} \n" \
                   f"Epoch run time was: {time_to_run} \n" \
                   f"Save status: {save_status} \n" \
                    "---------------------------------- \n" \
                   )
            

        test_loss, test_mAP = self.tester.validate(test_dl, device, 'Test')
        
        print(f"\nTest statistics \n" \
                   "---------------------------------- \n" \
                   f"Average test loss: {test_loss:.4f} \n" \
                   f"Average test mAP: {test_mAP:.4f} \n" \
                    "---------------------------------- \n" \
                   )


