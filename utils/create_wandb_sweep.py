import wandb
import yaml
import argparse
import os

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('--project', type=str, help='Id of the sweep', default="PoseEstimation6D")
        parser.add_argument('--wb', type=str, help='wb key', default="")
        args = parser.parse_args()

        with open('config/sweep_config.yaml') as f:
                sweep_conf = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_conf, 
                        entity="fantastic_4_0",
                        project=args.project)
        
        if not os.path.exists("./wandb_api_key.txt") and args.wb != "":
                with open("./wandb_api_key.txt", "w") as file:
                        file.write(args.wb)


        print(sweep_id)