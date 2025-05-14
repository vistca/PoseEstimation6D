from tqdm import tqdm
import time
import statistics
import torch

class Trainer():

    def __init__(self, model, optimizer, wandb_instance, epochs):
        self.model = model
        self.optimizer = optimizer
        self.wandb_instance = wandb_instance
        self.epochs = epochs
        self.checkpoint_path = '/checkpoints'

    def train(self, dataloader, device):
        
        train_losses = []

        for epoch in range(self.epochs):

            self.model.train()
            total_loss = 0.0
            nr_batches = 0

            loss_classifier = 0
            loss_box_reg = 0
            loss_objectness = 0
            loss_rpn_box_reg = 0
            timings = {"DL update iter" : [], "load" : [], "fit/loss" : [], "backprop" : []}

            progress_bar = tqdm(dataloader, desc="Training", ncols=100)
            start = time.perf_counter()
            for batch_idx, batch in enumerate(progress_bar):
                end = time.perf_counter()
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
                print(device.type)
                #with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = self.model(images, targets)

                loss_classifier += loss_dict["loss_classifier"].item()
                loss_box_reg += loss_dict["loss_box_reg"].item()
                loss_objectness += loss_dict["loss_objectness"].item()
                loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()
                loss = sum(loss for loss in loss_dict.values())

                end = time.perf_counter()
                timings["fit/loss"].append(end - start)

                start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                #self.optimizer.zero_grad()
                
                #or 
                #for param in model.parameters():
                #param.grad = None
                # scaler = torch.amp.GradScaler()
                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()

                loss.backward()
                self.optimizer.step()
                end = time.perf_counter()
                timings["backprop"].append(end - start)

                # Track loss
                total_loss += loss.item()

                nr_batches += 1

                progress_bar.set_postfix(total=total_loss/nr_batches, class_loss=loss_classifier/nr_batches, box_reg=loss_box_reg/nr_batches)

                #if nr_batches == 2:
                #    break

            avg_loss = total_loss / len(dataloader)
            train_losses.append(avg_loss)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

            self.wandb_instance.log_metric({"Training total_loss" : avg_loss,
                                            "Training class_loss" : loss_classifier / len(dataloader),
                                            "Training box_loss" : loss_box_reg / len(dataloader),
                                            "Training background_loss" : loss_objectness / len(dataloader),
                                            "Training rpn_box_loss" : loss_rpn_box_reg / len(dataloader)
                                            })
            

            self.wandb_instance.log_metric({
                                            "DL update iter" : statistics.mean(timings["DL update iter"]),
                                            "Time load_data" : statistics.mean(timings["load"]),
                                            "Time fit/calc_loss" : statistics.mean(timings["fit/loss"]),
                                            "Time backprop" : statistics.mean(timings["backprop"]),
                                        })
        
        #Add if statement here with some kind of input variable for toggle

        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.checkpoint_path)
    
