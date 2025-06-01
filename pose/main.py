import sys
import os

# Get the path to the parent directory (i.e., project_root/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add it to sys.path
sys.path.insert(0, parent_dir)

from utils.wandb_setup import WandbSetup
import yaml
import torch
from utils.optimizer_loader import OptimLoader
from timm.data.loader import MultiEpochsDataLoader
from prep_data import download_data, yaml_to_json, transfer_data
from .data.pose_dataset import PoseDataset
import os
from train_test_handler import TTH
from pose.train import Trainer
from pose.test import Tester
from utils.runtime_args import add_runtime_args
from utils.scheduler_loader import ScheduleLoader
from pose.models.model_creator import create_model

def run_program(args):
    dataset_root = args.data + "/Linemod_preprocessed"
    runtime_dir_path = os.path.dirname(os.path.abspath(__file__))

    wandb_instance = WandbSetup(args, "PosePhase3")

    if args.ld != "" and not os.path.exists(dataset_root):
        download_data(args.ld, args.data)
        yaml_to_json(args.data + "Linemod_preprocessed/data/")
        transfer_data("./datasets/Linemod_preprocessed/models/", "models_info")


    if args.lm != "":
        try:
            load_path = f"{runtime_dir_path}/checkpoints/{args.lm}.pt"
            model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
            print("Model loaded")
        except:
             raise("Could not load the model, might be due to missmatching models or something else")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    selected_model = create_model(args.mod)
    model = selected_model.to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    optimloader = OptimLoader(args.optimizer, model_params, args.lr)
    optimizer = optimloader.get_optimizer()

    schedulerloader = ScheduleLoader(optimizer, args.scheduler, args.bs, 9479)
    scheduler = schedulerloader.get_scheduler()

    trainer = Trainer(model, optimizer, scheduler)
    tester = Tester(model)

    tth = TTH(model,optimizer, 
              wandb_instance, args.epochs,
              trainer, tester, runtime_dir_path)


    with open('config/global_runtime_config.yaml') as f:
            config_dict = yaml.safe_load(f)

    split_percentage = {
                        "train_%" : config_dict["train_%"],
                        "test_%" : config_dict["test_%"],
                        "val_%" : config_dict["val_%"],
                        }

    
    train_dataset = PoseDataset(dataset_root, split_percentage, model.get_dimension(), split="train")
    test_dataset = PoseDataset(dataset_root, split_percentage, model.get_dimension(), split="test")
    val_dataset = PoseDataset(dataset_root, split_percentage, model.get_dimension(), split="val")
    
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=args.bs, 
                                         shuffle=True, num_workers=args.w)
    
    val_loader = MultiEpochsDataLoader(val_dataset, batch_size=args.bs, 
                                       shuffle=True, num_workers=args.w)
    
    test_loader = MultiEpochsDataLoader(test_dataset, batch_size=args.bs, 
                                        shuffle=True, num_workers=args.w)


    tth.train_test_val_model(train_loader, val_loader, test_loader,
                             device, args.sm, args.test)
    

if __name__ == "__main__":
    args = add_runtime_args()
    run_program(args)
