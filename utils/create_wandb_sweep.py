import wandb
import yaml
import argparse

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('--project', type=str, help='Id of the sweep', default="PoseEstimation6D")
        args = parser.parse_args()

        with open('config/sweep_config.yaml') as f:
                sweep_conf = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_conf, 
                        entity="fantastic_4_0",
                        project=args.project)

        print(sweep_id)