

import wandb
import os

class WandbSetup():
    def __init__(self, args, project_name):
        os.environ["WANDB_SILENT"] = "true"
        self.logging = args.log
        
        if self.logging == True:
            if args.wb == "":
                try:
                    file = open("wandb_api_key.txt").readlines()
                    for lines in file:
                        args.wb = lines
                except:
                    raise("Login to wandb failed, a key was not provided")
            else:
                try:
                    wandb.login(key=args.wb)
                except:
                    raise("Login to wandb failed, provided key is invalid")
                
            self.run = wandb.init(
                entity="fantastic_4_0",
                project=project_name,
                config=args,  
            )
            print("Login complete")
        else:
            print("Login disabled")


    def log_metric(self, log_dict):
        if self.logging:
            self.run.log(log_dict)

    def get_run_name(self):
        return self.run.name

    def log_hist(self, key, value):
        if self.logging:
            self.run.log({key : wandb.Histogram(value)})

    def log_line_plot(self, main_key, epoch, dict_content):
        if self.logging:
            keys = []
            values = []

            
            table = wandb.Table(columns=["values"])

            for key, value in dict_content.items():
                table.add_data(value)
                keys.append(key)
                values.append(value)

                self.run.log({
                        main_key : wandb.plot.line_series( 
                        xs=[epoch],
                        ys=values,
                        keys=keys,
                        title=main_key,
                        xname="Epochs"
                    )
                })
                