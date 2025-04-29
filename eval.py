
# We should create the test/eval function to be able to use it for both testing and validation. And the validation maybe should be added connected to the training loop

'''
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

            #if (nr_batches >= 500):
            #  break

    val_loss = val_loss / nr_batches #len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy
'''
