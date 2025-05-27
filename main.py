from utils.wandb_setup import WandbSetup
import yaml
import torch
from utils.optimizer_loader import OptimLoader
from utils.scheduler_loader import ScheduleLoader
from models.fasterRCNN import FasterRCNN
from timm.data.loader import MultiEpochsDataLoader
from prep_data import download_data, yaml_to_json
from data.faster_dataset import FasterDataset
import os
from train_test_handler import TTH
from test import Tester
from train import Trainer
from utils.runtime_args import add_runtime_args

def run_program(args):
    dataset_root = args.data + "/Linemod_preprocessed"

    wandb_instance = WandbSetup("testround", args, "PoseEstimation6D")

    if args.ld != "" and not os.path.exists(dataset_root):
        download_data(args.ld, args.data)
        yaml_to_json(args.data + "Linemod_preprocessed/data/")

    model = FasterRCNN(args.tr, args.fm)

    if args.lm != "":
        try:
            model.get_model().load_state_dict(torch.load('checkpoints/'+args.lm + ".pt", weights_only=True))
            print("Model loaded")
        except:
             raise("Could not load the model, might be due to missmatching models or something else")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.get_model().to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]

    optimloader = OptimLoader(args.optimizer, model_params, args.lr)
    optimizer = optimloader.get_optimizer()

    schedulerloader = ScheduleLoader(optimizer, args.scheduler)
    scheduler = schedulerloader.get_scheduler()

    trainer = Trainer(model, optimizer, wandb_instance, scheduler)
    tester = Tester(model, wandb_instance)

    tth = TTH(model,optimizer, 
              wandb_instance, args.epochs,
              trainer, tester
              )

    dataset_root = args.data + "/Linemod_preprocessed"

    with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

    split_percentage = {
                        "train_%" : config_dict["train_%"],
                        "test_%" : config_dict["test_%"],
                        "val_%" : config_dict["val_%"],
                        }

    
    train_dataset = FasterDataset(dataset_root, split_percentage, split="train")
    test_dataset = FasterDataset(dataset_root, split_percentage, split="test")
    val_dataset = FasterDataset(dataset_root, split_percentage, split="val")
    
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=args.bs, 
                                         shuffle=True, num_workers=args.w)
    
    val_loader = MultiEpochsDataLoader(val_dataset, batch_size=args.bs, 
                                       shuffle=True, num_workers=args.w)
    
    test_loader = MultiEpochsDataLoader(test_dataset, batch_size=args.bs, 
                                        shuffle=True, num_workers=args.w)


    tth.train_test_val_model(train_loader, val_loader, test_loader,
                             device, args.sm)


if __name__ == "__main__":
    
    args = add_runtime_args()
    run_program(args)
