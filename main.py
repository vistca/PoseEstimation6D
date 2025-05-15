from utils.wandb_setup import WandbSetup
import argparse
import yaml
import torch
from train import Trainer
from test import Tester
from utils.optimizer_loader import OptimLoader
from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from models.fasterRCNN import FasterRCNN
from models.yolo import Yolo
from timm.data.loader import MultiEpochsDataLoader
from prep_data import download_data, yaml_to_json
from data.faster_dataset import FasterDataset
import os

def run_program(parser):
    parsed_args = parser.parse_args()
    dataset_root = parsed_args.data + "/Linemod_preprocessed"

    wandb_instance = WandbSetup("testround", parsed_args)

    if parsed_args.ld != "" and not os.path.exists(dataset_root):
        download_data(parsed_args.ld, parsed_args.data)
        yaml_to_json(parsed_args.data + "Linemod_preprocessed/data/")

    model = FasterRCNN()#ModelLoader(parsed_args.head, parsed_args.backbone)
    #modelloader = Yolo()

    if parsed_args.lm != "":
        try:
            model.load_state_dict(torch.load('checkpoints/'+parsed_args.lm + ".pt", weights_only=True))
        except:
             raise("Could not load the model, might be due to missmatching models")


    #model = modelloader.get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]

    optimloader = OptimLoader(parsed_args.optimizer, model_params, parsed_args.lr)
    optimizer = optimloader.get_optimizer()

    trainer = Trainer(model, optimizer, wandb_instance, parsed_args.epochs)
    tester = Tester(model, wandb_instance)

    dataset_root = parsed_args.data + "/Linemod_preprocessed"
    
    train_dataset = FasterDataset(dataset_root, split="train")
    test_dataset = FasterDataset(dataset_root, split="test")
    val_dataset = FasterDataset(dataset_root, split="val")
    
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=parsed_args.bs, shuffle=True, num_workers=parsed_args.w)
    test_loader = MultiEpochsDataLoader(test_dataset, batch_size=parsed_args.bs, shuffle=True, num_workers=parsed_args.w)
    val_loader = MultiEpochsDataLoader(val_dataset, batch_size=parsed_args.bs, shuffle=True, num_workers=parsed_args.w)

    print("done with loading")
    trainer.train(train_loader, val_loader, device)


    torch.save()


    print("testing phase")
    tester.validate(test_loader, device)

    if parsed_args.sm != "":
        torch.save(model.state_dict(), 'checkpoints/' + parsed_args.sm)
    

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
    
    parser.add_argument('--log', type=bool, action=argparse.BooleanOptionalAction,
                        help='If run should be logged using wandb', default=True)
    
    parser.add_argument('--w', type=int,
                        help='The number of workers that should be used', default=2)
    
    parser.add_argument('--lm', type=str,
                        help='The name of the model that is to be loaded', default="")
    
    parser.add_argument('--sm', type=str,
                        help='The name of the model that is to be saved', default="")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    add_runtime_args(parser)
    run_program(parser)
