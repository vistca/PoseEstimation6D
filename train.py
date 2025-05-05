import tqdm

class Trainer():

    def __init__(self, model, optimizer, wandb_instance):
        self.model = model
        self.optimizer = optimizer
        self.wandb_instance = wandb_instance

    def train(self, dataloader, loss_fn):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        nr_batches = 0
        for (inputs, targets) in tqdm(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            pred = self.model(inputs)
            loss = loss_fn(pred, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            nr_batches += 1

        train_loss = running_loss / nr_batches #len(dataloader)
        train_accuracy = 100. * correct / total
        print(f'Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

        self.wandb_instance.log_metric({"training_loss" : train_loss, "training_accuracy" : train_accuracy})


