from torch.utils.data import DataLoader
from pose2.pose_dataset import PoseDataset
from pose2.models.bb8_2 import BB8Model_2
from tqdm import tqdm
from torch import nn
import torch
import os

def train_model(batch_size = 128, num_epochs=10):
    # Configuration
    dataset_root = "./datasets/Linemod_preprocessed/" # Ensure this path is correct relative to your Colab environment
    learning_rate = 0.001
    checkpoint_dir = "./pose2/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    split_percentage = {
                        "train_%" : 0.6,
                        "test_%" : 0.2,
                        "val_%" : 0.2,
                        }

    # Model and optimizer
    model = BB8Model_2().to(device) #.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error Loss for keypoint regression

    # Dataset and DataLoader
    train_dataset = PoseDataset(dataset_root, split_percentage, model.get_dimension(), split="train")
    val_dataset = PoseDataset(dataset_root, split_percentage, model.get_dimension(), split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)


    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        train_loss = 0.0

        # Training loop with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images = batch['rgb'].to(device) #.cuda()
            targets = batch['points_2d'].to(device) #.cuda()

            pred_points = model(images) # Forward pass: predict 2D points and ignore symmetry output

            # Removed the redundant loop for selected_preds
            loss = criterion(pred_points, targets) # Calculate loss between predicted and target points
            optimizer.zero_grad() # Zero the gradients before backpropagation
            loss.backward() # Perform backpropagation
            optimizer.step() # Update model parameters
            train_loss += loss.item() # Accumulate training loss

            break

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                images = batch['rgb'].to(device) #.cuda()
                targets = batch['points_2d'].to(device) #.cuda()

                pred_points = model(images) # Forward pass

                # Removed the redundant loop for selected_preds
                loss = criterion(pred_points, targets) # Calculate validation loss
                val_loss += loss.item() # Accumulate validation loss

                break

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # 🔄 Saving "last" checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, os.path.join(checkpoint_dir, "last.pth"))

        # ⭐ Saving "best" checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(checkpoint_dir, "best.pth"))
            print("✅ New best model saved.")

    return model


if __name__ == '__main__':

    # --- Main execution block (for demonstration) ---
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    data_dir = "Linemod_preprocessed/" # Make sure this path is correct for your dataset
    batch_size = 128


    # Step 1: Train the model
    print(f"Starting model training...")
    trained_model = train_model(batch_size=batch_size, num_epochs=5)
    print(f"Training complete.")


