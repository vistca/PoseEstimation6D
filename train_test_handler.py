import time
import datetime
import torch
import os

class TTH():
    def __init__(self, model, optimizer, wandb_instance, epochs, trainer, tester, runtime_path):
        self.model = model
        self.optimizer = optimizer
        self.wandb = wandb_instance
        self.epochs = epochs
        self.trainer = trainer
        self.tester = tester
        self.runtime_path = runtime_path

    def save_model(self, path):
        if os.path.exists(path):
            os.remove(path)
        save_path = f"{self.runtime_path}/checkpoints/{path}.pt"
        torch.save(self.model.state_dict(), save_path)
        return "Model saved"
        
    
    def print_info(self, header, info_dict):
        print(header)
        print("----------------------------------")
        for key, value in info_dict.items():
            print(f"{key}: {value}")
        print("----------------------------------\n")


    def train_test_val_model(self, train_dl, val_dl, test_dl, device, save_path, run_test):
        start = time.perf_counter()

        best_comp_metric = None

        for epoch in range(self.epochs):
            
            print(f"\nStarting epoch {epoch+1}/{self.epochs}")

            train_output = self.trainer.train_one_epoch(train_dl, device)

            val_output, comp_metric = self.tester.validate(val_dl, device, 'Validation')
            self.wandb.log_metric({**val_output, **train_output})
            
            end = time.perf_counter()
            time_diff = end - start
            time_to_run = str(datetime.timedelta(seconds=time_diff))

            save_status = "Model not saved"
            if not best_comp_metric or comp_metric <= best_comp_metric:
                best_comp_metric = comp_metric

                if save_path != "":
                    save_status = self.save_model(save_path)
                    
                


            train_output.update(val_output)
            train_output.update({
                "Epoch run time was" : time_to_run, 
                "Save status" : save_status
            })


            self.print_info(
                f"\nEpoch statistics for epoch {epoch+1}/{self.epochs}", 
                train_output
            )

            start = time.perf_counter()   

        self.wandb.log_metric({"Best validation metric" : best_comp_metric})

        if run_test:

            test_output, _ = self.tester.validate(test_dl, device, 'Test')

            self.wandb.log_metric(test_output)
            
            self.print_info(
                    f"\nTest statistics", 
                    test_output
                )
