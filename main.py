from utils.wandb_setup import WandbSetup
import argparse
import yaml
from models.model_loader import ModelLoader
import torch
from train import Trainer
from utils.optimizer_loader import OptimLoader
from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
import subprocess
import os

def download_data(google_folder, dataset_root):
    os.mkdir(f"./{dataset_root}/")
    subprocess.run(["gdown", "--folder", google_folder, "-O", f"PoseEstimation6D/{dataset_root}/"],
        check=True)

def run_program(parser):
    parsed_args = parser.parse_args()

    wandb_instance = WandbSetup("round5", parsed_args)

    model = ModelLoader(parsed_args.head, parsed_args.backbone)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)
        print("Model is running on gpu")
    else:
        device = torch.device("cpu")
        print("Model is running on cpu")

    optimizer = OptimLoader(parsed_args.optimizer, model.parameters(), parsed_args.lr)
    trainer = Trainer(model, optimizer, wandb_instance)
    
    dataset_root = parsed_args.data
    
    if parser.ld != "":
        download_data(parser.lr, dataset_root)


    train_dataset = CustomDataset(dataset_root, split="train")
    test_dataset = CustomDataset(dataset_root, split="test")
    train_loader = DataLoader(train_dataset, batch_size=parsed_args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=parsed_args.bs, shuffle=False)

def add_runtime_args(parser):
    with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

    parser.add_argument('--lr', type=float,
                    help='The learning rate', default=config_dict['learning_rate'])
    
    parser.add_argument('--bs', type=int,
                    help='The bastch size', default=config_dict['batch_size'])
    
    parser.add_argument('--epochs', type=int,
                    help='The number of epochs', default=config_dict['epochs'])
    
    parser.add_argument('--dropout', type=float,
                    help='The percentage of dropout', default=config_dict['dropout'])
    
    parser.add_argument('--optimizer', type=str,
                    help='The optimizer to use', default=config_dict['optimizer'])
    
    parser.add_argument('--backbone', type=str,
                    help='The name of the backbone', default=config_dict['backbone'])

    parser.add_argument('--head', type=str,
                    help='The name of the head', default=config_dict['head'])
    
    parser.add_argument('--data', type=str,
                    help='The input folder', default=config_dict['data_dir'])
    
    parser.add_argument('--ld', type=str,
                    help='Google drive download path', default="")
    
    try:
        file = open("wandb_api_key.txt").readlines()
        for lines in file:
            api_key = lines
    except:
        api_key = ""

    parser.add_argument('--wb', type=bool,
                        help='If data is available locally or should be downloaded', default=api_key)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    add_runtime_args(parser)
    run_program(parser)
