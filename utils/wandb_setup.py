import random

import wandb

class WandbSetup():
    def __init__(self, name_of_round, parsed_args):
        try:
            file = open("wandb_api_key.txt").readlines()
            for lines in file:
                api_key = lines
        except:
            api_key = input("Give me your key for wandb :) \n - ")

        wandb.login(key=api_key)
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


    def log_metric(self, log_dict):
        self.run.log(log_dict)



        
     
