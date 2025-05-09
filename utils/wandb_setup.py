

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

    def log_hist(self, key, value):
        self.run.log({key : wandb.Histogram(value)})

    def log_line_plot(self, main_key, epoch, dict_content):
        keys = []
        values = []

        
        table = wandb.Table(columns=["values"])

        for key, value in dict_content.items():
           table.add_data(value)
           keys.append(key)
           values.append(value)


        # self.run.log({
        #         main_key : wandb.plot.line( 
        #         table=table,
        #         x="epochs",
        #         y="time"
        #     )
        # })

        self.run.log({
                main_key : wandb.plot.line_series( 
                xs=[epoch],
                ys=values,
                keys=keys,
                title=main_key,
                xname="Epochs"
            )
        })


        # table: wandb.Table,
        # x: str,
        # y: str,
        # stroke: str | None = None,
        # title: str = "",
        # split_table: bool = False,
        



        
     
