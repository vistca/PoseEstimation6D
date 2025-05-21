from tqdm import tqdm
import time
import statistics
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Tester:

    def __init__(self, model, wandb_instance):
        self.model = model
        self.wandb_instance = wandb_instance
        self.loss_fn = torch.nn.MSELoss()


    def validate(self, dataloader, device, type):

        self.model.eval()
        val_loss = 0.0

        print(f"Starting {type}...")
        progress_bar = tqdm(dataloader, desc=type, ncols=100)

        with torch.no_grad():
            for batch_id, batch in enumerate(progress_bar):
                images = batch["rgb"].to(device)
                targets = []
                for i in range(images.shape[0]):
                    target = {
                        "boxes": batch["bbox"][i].to(device).unsqueeze(0),
                        "labels": batch["obj_id"][i].to(device).long().unsqueeze(0)
                    }
                    targets.append(target)

                # Forward pass

                # weird quirk with eval() only returning predictions. Probably
                # bad practice to elvaluate in train() mode.

                # doing it like this takes forever, might need to check this and update it accordingly

                self.model.train()
                loss_dict = self.model(images, targets)
                self.model.eval()

                #print(type(loss_dict), loss_dict)  # Debugging output
                loss = sum(loss for loss in loss_dict.values())

                val_loss += loss.item()
                loss_classifier += loss_dict["loss_classifier"].item()
                loss_box_reg += loss_dict["loss_box_reg"].item()
                loss_objectness += loss_dict["loss_objectness"].item()
                loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

                # Inference for mAP
                outputs = self.model(images)

                preds = []
                gts = []
                for pred, tgt in zip(outputs, targets):
                    preds.append({
                        "boxes": pred["boxes"].cpu(),
                        "scores": pred["scores"].cpu(),
                        "labels": pred["labels"].cpu()
                    })
                    gts.append({
                        "boxes": tgt["boxes"].cpu(),
                        "labels": tgt["labels"].cpu()
                    })

                progress_bar.set_postfix(total=val_loss/(batch_id + 1), 
                                         class_loss=loss_classifier/(batch_id + 1), 
                                         box_reg=loss_box_reg/(batch_id + 1))

        avg_loss = val_loss / len(dataloader)

        # Log to wandb
        self.wandb_instance.log_metric({
            f"{type} total_loss": avg_loss,
        })

        return round(avg_loss, 4)
    
