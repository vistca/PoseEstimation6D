from tqdm import tqdm
import time
import statistics
import torch
from torch.amp import autocast, GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Trainer():

    def __init__(self, model, optimizer, wandb_instance, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.wandb_instance = wandb_instance
        self.scheduler = scheduler

    def train_one_epoch(self, train_loader, device):

        metric = MeanAveragePrecision()

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
            self.model.train()
            timings["DL update iter"].append(end - start)
            
            start = time.perf_counter()
            images = batch["rgb"].to(device)
            targets = []

            # We must be able to improve/remove this loop
            for i in range(images.shape[0]):
                target = {}
                target["boxes"] = batch["bbox"][i].to(device).unsqueeze(0)  # Add batch dimension
                target["labels"] = batch["obj_id"][i].to(device).long().unsqueeze(0)  # Add batch dimension
                targets.append(target)
            
            end = time.perf_counter()
            timings["load"].append(end - start)

            start = time.perf_counter()
            # Using mixed precision training
            if device.type == 'cuda':
                with autocast(device.type):
                    loss_dict = self.model(images, targets)
            else:
                loss_dict = self.model(images, targets)
            
            # self.model.eval()
            # outputs = self.model(images)
            # preds = []
            # gts = []
            # for pred, tgt in zip(outputs, targets):
            #     preds.append({
            #         "boxes": pred["boxes"].cpu(),
            #         "scores": pred["scores"].cpu(),
            #         "labels": pred["labels"].cpu()
            #     })
            #     gts.append({
            #         "boxes": tgt["boxes"].cpu(),
            #         "labels": tgt["labels"].cpu()
            #     })

            # metric.update(preds, gts)
              

            loss_classifier += loss_dict["loss_classifier"].item()
            loss_box_reg += loss_dict["loss_box_reg"].item()
            loss_objectness += loss_dict["loss_objectness"].item()
            loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()
            loss = sum(loss for loss in loss_dict.values())

            end = time.perf_counter()
            timings["fit/loss"].append(end - start)

            start = time.perf_counter()
            
            # Added parts so we only decrease the learning rate if the loss was not scaled.
            # If the loss was scaled then it isn't any reason to reduce it since it will
            # not be representative of the actual decrease
            scaler.scale(loss).backward()
            scaled_factor = scaler.get_scale()
            scaler.step(self.optimizer)
            scaler.update()
            if not scaled_factor <= scaler.get_scale() and self.scheduler:
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
                #self.wandb_instance.log_metric({"Learning rate after batch" : self.scheduler._last_lr})

            end = time.perf_counter()
            timings["backprop"].append(end - start)

            # Track loss
            total_loss += loss.item()

            progress_bar.set_postfix(total=total_loss/(batch_id + 1), 
                                     class_loss=loss_classifier/(batch_id + 1), 
                                     box_reg=loss_box_reg/(batch_id + 1))
            

        # Inference for mAP

        #val_metrics = metric.compute()

        avg_loss = total_loss / len(train_loader)


        # self.wandb_instance.log_metric({"Training total_loss" : avg_loss,
        #                                 "Training class_loss" : loss_classifier / len(train_loader),
        #                                 "Training box_loss" : loss_box_reg / len(train_loader),
        #                                 "Training background_loss" : loss_objectness / len(train_loader),
        #                                 "Training rpn_box_loss" : loss_rpn_box_reg / len(train_loader)
        #                                 })
        

        # self.wandb_instance.log_metric({
        #                                 "DL update iter" : statistics.median(timings["DL update iter"]),
        #                                 "Time load_data" : statistics.median(timings["load"]),
        #                                 "Time fit/calc_loss" : statistics.median(timings["fit/loss"]),
        #                                 "Time backprop" : statistics.median(timings["backprop"]),
        #                             })
    
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
    
