import random
#from config import config

import wandb

class Wandb():
    def __init__(self):

        try:
            file = open("wandb_api_key.txt").readlines()
            for lines in file:
                api_key = lines
        except:
            api_key = input("Give me your key for wandb :) \n - ")

        print(api_key)
        wandb.login(key=api_key)
        # # Start a new wandb run to track this script.
        # self.run = wandb.init(
        #     # Set the wandb entity where your project will be logged (generally your team name).
        #     entity="fantastic_4_0",
        #     # Set the wandb project where this run will be logged.
        #     project="PoseEstimation6D",

        #     # Track hyperparameters and run metadata.
        #     config={
        #        "model": "setup_test",
        #     },
        # )

    def log_metric(self, log_name, log_value):
        self.run.log({log_name, log_value})
