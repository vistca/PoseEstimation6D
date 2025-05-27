from pose.train import Trainer
from pose.test import Tester
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
        self.trainer = Trainer(model, optimizer, epochs)
        self.tester = Tester(model)

    def save_model(self, path, prev_losses, curr_loss):
        if len(prev_losses) == 0 or curr_loss < min(prev_losses):
            if os.path.exists(path):
                os.remove(path)
            torch.save(self.model.state_dict(), 'checkpoints/'+ path + ".pt")
            return "Model saved"
        return "Model not saved"

    def train_test_val_model(self, train_dl, val_dl, test_dl, device, save_path):
        start = time.perf_counter()

        losses = {"train_loss" : [], "val_loss" : []}

        for epoch in range(self.epochs):
            
            print(f"\nStarting epoch {epoch+1}/{self.epochs}")

            train_result = self.trainer.train_one_epoch(train_dl, device)
            train_avg_loss = train_result["Training total_loss"]
            

            val_result = self.tester.validate(val_dl, device, 'Validation')
            val_avg_loss = val_result["Validation total_loss"]
            

            result_dict= {**train_result, **val_result}
            self.wandb.log_metric(result_dict)
            
            end = time.perf_counter()
            time_diff = end - start
            time_to_run = str(datetime.timedelta(seconds=time_diff))

            save_status = "Unapplicable"
            if save_path != "":
                save_status = self.save_model(save_path, 
                                              losses["val_loss"],
                                              val_avg_loss)
            
            losses["train_loss"].append(train_avg_loss)
            losses["val_loss"].append(val_avg_loss)

            print(f"Epoch statistics for epoch {epoch+1}/{self.epochs} \n" \
                   "---------------------------------- \n" \
                   f"Average training loss: {train_avg_loss:.4f} \n" \
                   f"Average validation loss: {val_avg_loss:.4f} \n" \
                   f"Epoch run time was: {time_to_run} \n" \
                   f"Save status: {save_status} \n" \
                    "---------------------------------- \n" \
                   )
            
        test_result = self.tester.validate(test_dl, device, 'Test')
        test_loss = test_result["Test total_loss"]
        self.wandb.log_metric({
            "Test total_loss": test_loss,
        })

        print(f"\nTest statistics \n" \
                   "---------------------------------- \n" \
                   f"Average test loss: {test_loss:.4f} \n" \
                    "---------------------------------- \n" \
                   )


