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
import shutil
from models.fasterRCNN import FasterRCNN
from models.yolo import Yolo


def download_data(google_folder, dataset_root):
    output_path = dataset_root + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        subprocess.run(["gdown", "--folder", str(google_folder), "-O", "tmp/"],
            check=True)
        subprocess.run(["unzip", "tmp/DenseFusion/Linemod_preprocessed.zip", "-d", output_path],
        check=True 
        )
        shutil.rmtree('tmp')

def tmploss():
     return 1

def run_program(parser):
    parsed_args = parser.parse_args()

    wandb_instance = WandbSetup("testround", parsed_args)

    if parsed_args.ld != "":
        download_data(parsed_args.ld, parsed_args.data)

    modelloader = FasterRCNN()#ModelLoader(parsed_args.head, parsed_args.backbone)
    #modelloader = Yolo()
    model = modelloader.get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]

    optimloader = OptimLoader(parsed_args.optimizer, model_params, parsed_args.lr)
    optimizer = optimloader.get_optimizer()

    trainer = Trainer(model, optimizer, wandb_instance, 1)

    dataset_root = parsed_args.data + "/Linemod_preprocessed"
    train_dataset = CustomDataset(dataset_root, split="train")
    test_dataset = CustomDataset(dataset_root, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=parsed_args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=parsed_args.bs, shuffle=False)

    print("done with loading")
    trainer.train(train_loader, device)

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

    parser.add_argument('--wb', type=str,
                        help='If data is available locally or should be downloaded', default="")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    add_runtime_args(parser)
    run_program(parser)
