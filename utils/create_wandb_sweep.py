import wandb
import yaml
import argparse
import os
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, help='project in which the sweep is to be located', default="PoseEstimation6D")
    parser.add_argument('--wb', type=str, help='wb key', default="")
    parser.add_argument('--sf', type=str, help='file location of sweep', default="2dBox/config/fasterrcnn_sweep_config")
    args = parser.parse_args()
    
    
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
    

    try:
        with open(args.sf + ".yaml") as f:
           sweep_conf = yaml.safe_load(f)
    except:
        raise("File path is unvalid")

    sweep_id = wandb.sweep(sweep_conf, 
                           entity="fantastic_4_0",
                           project=args.project)

    if not os.path.exists("./wandb_api_key.txt") and args.wb != "":
        with open("./wandb_api_key.txt", "w") as file:
           file.write(args.wb)


    print(sweep_id)