import wandb
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, help='Id of the sweep', default="")
    parser.add_argument('--count', type=int, help='The number of rounds', default=1)
    parser.add_argument('--project', type=str, help='The name of the project', default="PoseEstimation6D")
    parser.add_argument('--wb', type=str, help='wandb key', default="")

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


    wandb.agent(args.id, 
                count=args.count,
                entity="fantastic_4_0",
                project=args.project)

