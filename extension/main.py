
import os
import yaml
import torch
from .models.depth_nn import DepthNN
from .models.rgb_nn import RgbNN
from .models.combined_model import CombinedModel
from timm.data.loader import MultiEpochsDataLoader
from prep_data import download_data, yaml_to_json
#from .data.extension_dataset import ExtensionDataset
from pose2.pose_dataset import PoseEstDataset
from .test import Tester
from .train import Trainer

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

    model = CombinedModel(device, args.mod)

    # TODO: should look at this loading, does it load the entire model for pose or just the resnet part?
    if args.lm != "":
        try:
            #for name, model in models.items():
            #    load_path = f"{runtime_dir_path}/checkpoints/{args.lm}_{name}.pt"
            load_path = f"{runtime_dir_path}/checkpoints/{args.lm}.pt"
            model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
            print("Models loaded")
        except:
             raise("Could not load the model, might be due to missmatching models or something else")

    model_params = model.get_parameters()

    optimloader = OptimLoader(args.optimizer, model_params, args.lr)
    optimizer = optimloader.get_optimizer()

    schedulerloader = ScheduleLoader(optimizer, args.scheduler, args.bs, 9479)
    scheduler = schedulerloader.get_scheduler()

    trainer = Trainer(model, optimizer, args)#, wandb_instance, scheduler)
    tester = Tester(model, args.epochs)

    tth = TTH(model,optimizer, 
              wandb_instance, args.epochs,
              trainer, tester, runtime_dir_path)

    dataset_root = args.data + "/Linemod_preprocessed"

    with open('config/global_runtime_config.yaml') as f:
            config_dict = yaml.safe_load(f)

    split_percentage = {
                        "train_%" : config_dict["train_%"],
                        "test_%" : config_dict["test_%"],
                        "val_%" : config_dict["val_%"],
                        }

    
    train_dataset = PoseEstDataset(dataset_root, split_percentage, model.get_dimensions(), split="train")
    test_dataset = PoseEstDataset(dataset_root, split_percentage,  model.get_dimensions(), split="test")
    val_dataset = PoseEstDataset(dataset_root, split_percentage,  model.get_dimensions(), split="val")
    
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
