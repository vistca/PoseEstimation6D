from tqdm import tqdm
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Tester:

    def __init__(self, model, wandb_instance):
        self.model = model
        self.wandb_instance = wandb_instance


    def validate(self, dataloader, device, type):

        self.model.eval()
        val_loss = 0.0
        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0

        correct_labels = 0
        total_labels = 0

        metric = MeanAveragePrecision()

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

                # Has to set the model to train and then eval again in order to get both
                # loss and predictions. In train mode we get loss, in eval we get preds. 
                # This is due to the torchvision implementation
                self.model.train()
                loss_dict = self.model(images, targets)
                self.model.eval()

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

                    # Calculate label accuracy
                    pred_labels = pred["labels"].cpu()
                    gt_labels = tgt["labels"].cpu()
                    if len(pred_labels) > 0:  # If the model made a prediction
                        pred_label = pred_labels[0].item()
                        gt_label = gt_labels[0].item()
                        if pred_label == gt_label:
                            correct_labels += 1
                    total_labels += 1


                metric.update(preds, gts)
                progress_bar.set_postfix(total=val_loss/(batch_id + 1), 
                                         class_loss=loss_classifier/(batch_id + 1), 
                                         box_reg=loss_box_reg/(batch_id + 1))
                

        avg_loss = val_loss / len(dataloader)
        val_metrics = metric.compute()

        label_accuracy = correct_labels / total_labels if total_labels > 0 else 0


        return {
            f"Average {type} loss" : round(avg_loss, 4), 
            f"Average {type} mAP" : round(val_metrics["map"].item(), 4),
            f"{type} class_loss": loss_classifier / len(dataloader),
            f"{type} box_loss": loss_box_reg / len(dataloader),
            f"{type} background_loss": loss_objectness / len(dataloader),
            f"{type} label_accuracy (%)": round(label_accuracy * 100, 2), 

        }, round(avg_loss, 4)
