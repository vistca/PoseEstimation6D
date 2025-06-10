
import os
import yaml
import torch
from .models.fasterRCNN import FasterRCNN
from timm.data.loader import MultiEpochsDataLoader
from prep_data import download_data, yaml_to_json
from .data.faster_dataset import FasterDataset
from .test import Tester
from .train import Trainer

# import os
# import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent_dir)

from utils.wandb_setup import WandbSetup
from utils.optimizer_loader import OptimLoader
from utils.scheduler_loader import ScheduleLoader
from utils.runtime_args import add_runtime_args
from train_test_handler import TTH

def run_program(args):
    runtime_dir_path = os.path.dirname(os.path.abspath(__file__))

    dataset_root = args.data + "/Linemod_preprocessed"

    wandb_instance = WandbSetup(args, args.project)

    if args.ld != "" and not os.path.exists(dataset_root):
        download_data(args.ld, args.data)
        yaml_to_json(args.data + "Linemod_preprocessed/data/")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FasterRCNN(args.tr, args.fm)
    model = model.get_model()
    if args.lm != "":
        try:
            load_path = f"{runtime_dir_path}/checkpoints/{args.lm}.pt"
            model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
            # model.get_model().load_state_dict(torch.load(runtime_dir_path+'checkpoints/'+args.lm + ".pt", 
            #                                              weights_only=True, map_location=device.type))
            print("Model loaded")
        except:
             raise("Could not load the model, might be due to missmatching models or something else")

    model = model.to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]

    optimloader = OptimLoader(args.optimizer, model_params, args.lr)
    optimizer = optimloader.get_optimizer()

    schedulerloader = ScheduleLoader(optimizer, args.scheduler, 6, 10)
    scheduler = schedulerloader.get_scheduler()

    trainer = Trainer(model, optimizer, wandb_instance, scheduler)
    tester = Tester(model, wandb_instance)

    tth = TTH(model,optimizer, 
              wandb_instance, args.epochs,
              trainer, tester, "./bbox"
              )

    dataset_root = args.data + "/Linemod_preprocessed"

    with open('config/global_runtime_config.yaml') as f:
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

    if args.test == False:
        tth.train_test_val_model(train_loader, val_loader, test_loader,
                                device, args.sm, args.test)
    else:
         test_output = tester.validate(test_loader, device, 'Test')
         print("\nTest statistics", test_output)


if __name__ == "__main__":
    
    args = add_runtime_args()
    run_program(args)
