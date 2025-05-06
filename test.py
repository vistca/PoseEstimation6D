
import torch

def test(model, dataloader, loss_fn):
        model.eval()
        val_loss = 0

        correct, total = 0, 0

        with torch.no_grad():
            nr_batches = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                pred = model(inputs)
                loss = loss_fn(pred, targets)

                val_loss += loss.item()
                _, predicted = pred.max(1) #outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                nr_batches += 1

        val_loss = val_loss / nr_batches #len(val_loader)
        val_accuracy = 100. * correct / total

        return val_accuracy, val_loss

