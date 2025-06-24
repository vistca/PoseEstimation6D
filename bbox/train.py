from tqdm import tqdm
import time
import statistics
from torch.amp import autocast, GradScaler

class Trainer():

    def __init__(self, model, optimizer, wandb_instance, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.wandb_instance = wandb_instance
        self.scheduler = scheduler

    def train_one_epoch(self, train_loader, device):

        self.model.train()
        total_loss = 0.0
        scaler = GradScaler()

        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0
        timings = {"DL update iter" : [], "load" : [], "fit/loss" : [], "backprop" : []}

        progress_bar = tqdm(train_loader, desc="Training", ncols=100)
        start = time.perf_counter()
        for batch_id, batch in enumerate(progress_bar):
            end = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)

            timings["DL update iter"].append(end - start)
            
            start = time.perf_counter()
            images = batch["rgb"].to(device)
            targets = []

            for i in range(images.shape[0]):
                target = {}
                target["boxes"] = batch["bbox"][i].to(device).unsqueeze(0)
                target["labels"] = batch["obj_id"][i].to(device).long().unsqueeze(0)
                targets.append(target)
            
            end = time.perf_counter()
            timings["load"].append(end - start)

            start = time.perf_counter()
            if device.type == 'cuda':
                with autocast(device.type):
                    loss_dict = self.model(images, targets)
            else:
                loss_dict = self.model(images, targets)
              

            loss_classifier += loss_dict["loss_classifier"].item()
            loss_box_reg += loss_dict["loss_box_reg"].item()
            loss_objectness += loss_dict["loss_objectness"].item()
            loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()
            loss = sum(loss for loss in loss_dict.values())

            end = time.perf_counter()
            timings["fit/loss"].append(end - start)

            start = time.perf_counter()
            
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            end = time.perf_counter()
            timings["backprop"].append(end - start)

            # Track loss
            total_loss += loss.item()

            progress_bar.set_postfix(total=total_loss/(batch_id + 1), 
                                     class_loss=loss_classifier/(batch_id + 1), 
                                     box_reg=loss_box_reg/(batch_id + 1))
            

        avg_loss = total_loss / len(train_loader)
        self.scheduler.step()
        print("LR is: ", self.scheduler._last_lr[0])

        return {
            "Average training loss" : round(avg_loss, 4),
            "Average training mAP" : 1,
            "Training class_loss" : loss_classifier / len(train_loader),
            "Training box_loss" : loss_box_reg / len(train_loader),
            "Training background_loss" : loss_objectness / len(train_loader),
            "Training rpn_box_loss" : loss_rpn_box_reg / len(train_loader),
            "DL update iter" : statistics.median(timings["DL update iter"]),
            "Time load_data" : statistics.median(timings["load"]),
            "Time fit/calc_loss" : statistics.median(timings["fit/loss"]),
            "Time backprop" : statistics.median(timings["backprop"]),
        }
    
