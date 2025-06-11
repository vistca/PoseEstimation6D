from utils.runtime_args import add_runtime_args
from pose2.pose_dataset import PoseEstDataset
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from utils.optimizer_loader import OptimLoader
from pose2.utils.models_points import get_ply_files
from pose2.utils.data_reconstruction import reconstruct_3d_points_from_pred, rescale_pred
from pose2.utils.add_calc import compute_ADD
from pose2.models.model_creator import create_model
from pose2.train import Trainer
from pose2.test import Tester
from utils.wandb_setup import WandbSetup
from utils.scheduler_loader import ScheduleLoader
from train_test_handler import TTH



def run_program(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runtime_dir_path = os.path.dirname(os.path.abspath(__file__))
    wandb_instance = WandbSetup(args, "PosePhase3")

    save_name = args.sm
    if args.sweep:
         save_name = wandb_instance.get_run_name()

    # Configuration
    dataset_root = "./datasets/Linemod_preprocessed/" # Ensure this path is correct relative to your Colab environment
    batch_size = args.bs
    checkpoint_dir = "./pose2/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    with open('./config/global_runtime_config.yaml') as f:
            config_dict = yaml.safe_load(f)

    # Setup the model
    selected_model = create_model(args.mod)
    model = selected_model.to(device)

    split_percentage = {
                        "train_%" : config_dict["train_%"],
                        "test_%" : config_dict["test_%"],
                        "val_%" : config_dict["val_%"],
                        }
    
    train_dataset = PoseEstDataset(dataset_root, split_percentage, model.get_dimension(), split="train")
    test_dataset = PoseEstDataset(dataset_root, split_percentage, model.get_dimension(), split="test")
    val_dataset = PoseEstDataset(dataset_root, split_percentage, model.get_dimension(), split="val")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Data loaders completed")


    # optimizer and loss function
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimloader = OptimLoader(args.optimizer, model_params, args.lr)
    optimizer = optimloader.get_optimizer()

    if args.lm != "":
        try:
            load_path = f"{runtime_dir_path}/checkpoints/{args.lm}.pt"
            model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
            print("Model loaded")
        except:
                raise("Could not load the model, might be due to missmatching models or something else")

    schedulerloader = ScheduleLoader(optimizer, args.scheduler, 6, 10)
    scheduler = schedulerloader.get_scheduler()

    trainer = Trainer(model, optimizer, args, scheduler)
    tester = Tester(model, args)

    tth = TTH(model,optimizer, 
              wandb_instance, args.epochs,
              trainer, tester, runtime_dir_path)
    
    tth.train_test_val_model(train_loader, val_loader, test_loader,
                             device, save_name, args.test)

    # best_val_loss = float('inf')

    # for epoch in range(args.epochs):

    #     train_results = trainer.train_one_epoch(train_loader, device, epoch)

    #     val_results = tester.validate(val_loader, device, epoch, type="Val")

    #     avg_train_loss = train_results["Training total_loss"]
    #     avg_val_loss = val_results["Val total_loss"]
    #     avg_ADD = val_results["Val total_ADD"]

    #     print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    #     if avg_val_loss < best_val_loss and args.sm != "":
    #         best_val_loss = avg_val_loss
            
    #         if os.path.exists(args.sm):
    #             os.remove(args.sm)
    #         save_path = f"./pose2/checkpoints/{args.sm}.pt"
    #         torch.save(model.state_dict(), save_path)
            
    #         print("New best model saved.")

    #     scheduler.step()
    #     print(scheduler._last_lr[0])

    # if args.test:
    #     test_results = tester.validate(test_loader, device, epoch, type="Test")
    #     avg_test_loss = test_results["Test total_loss"]
    #     avg_test_ADD = test_results["Test total_ADD"]
    #     print(f"Average test loss : {avg_test_loss}")
    #     print(f"Average test ADD : {avg_test_ADD}")


if __name__ == "__main__":
    args = add_runtime_args()
    run_program(args)

# python -m pose2.main --lr 0.001 --bs 16 --epochs 3 --mod bb8_1 --test True --wb 2bb395cd06545f69d0a731a962237634b99915e2 --scheduler CosineAnnealingWarmRestarts


