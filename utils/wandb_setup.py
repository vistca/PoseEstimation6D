

import wandb
import os

class WandbSetup():
    def __init__(self, name_of_round, parsed_args):
        os.environ["WANDB_SILENT"] = "true"
        
        if parsed_args.wb == "":
            try:
                file = open("wandb_api_key.txt").readlines()
                for lines in file:
                    parsed_args.wb = lines
            except:
                raise("Login to wandb failed, a key was not provided")
        else:
            try:
                wandb.login(key=parsed_args.wb)
            except:
                raise("Login to wandb failed, provided key is invalid")
        # Start a new wandb run to track this script.
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="fantastic_4_0",
            # Set the wandb project where this run will be logged.
            project="PoseEstimation6D",
            name = name_of_round,
            # Track hyperparameters and run metadata.
            config={
               "backbone": parsed_args.backbone,
               "head": parsed_args.head,
               "learning rate" : parsed_args.lr,
               "epochs" : parsed_args.epochs,
               "batch size" : parsed_args.bs
            },  
        )
        print("Login complete")


    def log_metric(self, log_dict):
        self.run.log(log_dict)



        
     
