

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import yaml
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from functools import lru_cache
from tqdm import tqdm


# --- BB8Model (PoseModel) Class Definition ---
# This class is included here to make the code self-contained,
# as it was defined in a previous turn and is used by the training function.
class BB8Model(nn.Module):
    def __init__(self, num_objects=15):
        super().__init__()
        # We use ResNet18 as the backbone instead of VGG for greater efficiency
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # We remove the final fully connected layer

        # Head for predicting 2D points (8 corners * 2 coordinates)
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16)# 8 points * 2 coordinates
        )

        # Head for the symmetry classifier (4 ranges for approximately symmetrical objects)
        self.symmetry_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_objects * 4),
            nn.Sigmoid()
        )

        # Cache for 3D models (not used by BB8Model itself, but kept for potential future use)
        self.model_points_cache = {}

    def forward(self, x):
        features = self.backbone(x)
        bbox_pred = self.bbox_head(features)
        symmetry_pred = self.symmetry_head(features)
        return bbox_pred, symmetry_pred

    # Removed get_3d_bbox_points from BB8Model as it's now in LinemodDataset


# --- LinemodDataset Class Definition ---
# This class is included here to make the code self-contained,
# as it was defined in a previous turn and is used by the training function.
class LinemodDataset(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.samples = self.get_all_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}.")

        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed
        )

        self.samples = self.train_samples if split == 'train' else self.test_samples

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Cache for configurations and GT
        self.config_cache = {}
        self.models_info_cache = {}
        self.camera_config_cache = {}

        # Load the necessary data, but not directly in the constructor
        self.models_info = self.load_models_info()
        self.gt_cache = {}  # Initialize GT cache here
        self.points2d_cache = {} # Cache for projected 2D points (normalized)

        # Preload all GT data for efficiency
        self.load_all_gt()

    def load_models_info(self):
        """Loads model information with manual caching"""
        if 'models_info' in self.models_info_cache:
            return self.models_info_cache['models_info']

        obj_path = os.path.join(self.dataset_root, 'models', "models_info.yml")
        with open(obj_path, 'r') as f:
            models_info = yaml.load(f, Loader=yaml.FullLoader)

        # Store in cache
        self.models_info_cache['models_info'] = models_info
        return models_info

    def get_all_samples(self):
        samples = []
        for folder_id in range(1, 16):
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "rgb")
            if os.path.exists(folder_path):
                sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
                samples.extend([(folder_id, sid) for sid in sample_ids])
        return samples

    def load_gt_for_folder(self, folder_id):
        """Loads GT for a specific folder with manual caching"""
        if folder_id in self.gt_cache:
            return self.gt_cache[folder_id]

        pose_file = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "gt.yml")
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                gt_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            gt_data = {}

        # Store in cache
        self.gt_cache[folder_id] = gt_data
        return gt_data

    def load_all_gt(self):
        """Preloads all GT data with caching"""
        for folder_id in range(1, 16):
            self.load_gt_for_folder(folder_id)

    def load_camera_config(self, folder_id):
        """Loads camera configuration with manual caching"""
        if folder_id in self.camera_config_cache:
            return self.camera_config_cache[folder_id]

        cam_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", 'info.yml')
        with open(cam_path, 'r') as f:
            camera_config = yaml.load(f, Loader=yaml.FullLoader)

        # Store in cache
        self.camera_config_cache[folder_id] = camera_config
        return camera_config

    def load_config(self, folder_id):
        """Obtains camera and object configuration with manual caching"""
        if folder_id not in self.config_cache:
            cam = self.load_camera_config(folder_id)
            obj = self.models_info
            self.config_cache[folder_id] = (cam, obj)
        return self.config_cache[folder_id]

    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def load_normal_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return transforms.ToTensor()(img)

    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.gt_cache.get(folder_id, {})
        # Changed to use sample_id directly as an integer, as keys in gt.yml are typically integers
        # str_id = str(sample_id)  # Removed this line

        if sample_id not in pose_data: # Changed from str_id to sample_id
            print(f"id: {sample_id},{pose_data}") # Changed from str_id to sample_id
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        pose = pose_data[sample_id][0] # Changed from str_id to sample_id
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32)
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        bbox = np.array(pose['obj_bb'], dtype=np.float32)
        obj_id = pose['obj_id']

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return translation, rotation, bbox, obj_id

    # Moved from BB8Model to LinemodDataset for efficiency
    def _get_3d_bbox_points(self, obj_id):
        """Obtains the 3D bounding box points for an object from dataset's models_info"""
        if obj_id not in self.models_info_cache: # Use dataset's own models_info_cache
            # This case should ideally not happen if load_models_info is called in __init__
            self.load_models_info()

        # models_info is already loaded in __init__
        info = self.models_info[obj_id]
        x_size = info['size_x']
        y_size = info['size_y']
        z_size = info['size_z']

        half_x = x_size / 2
        half_y = y_size / 2
        half_z = z_size / 2

        points_3d = np.array([
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [half_x, half_y, half_z],
            [-half_x, half_y, half_z]
        ], dtype=np.float32)

        # Cache this if needed, but models_info is already cached
        return points_3d


    def get_3d_bbox_projection(self, obj_id, rotation, translation, camera_matrix):
        """Projects the 3D bounding box points into the 2D image"""
        # Call the helper method within the dataset itself
        points_3d = self._get_3d_bbox_points(obj_id)
        points_2d, _ = cv2.projectPoints(points_3d, rotation, translation,
                                        camera_matrix, None)
        return points_2d.squeeze()

    def normalize_points(self, points_2d, bbox):
        """Normalizes the 2D points with respect to the bounding box"""
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Normalize between 0 and 1 with respect to the bounding box
        points_norm = (points_2d - np.array([x_min, y_min])) / np.array([width, height])
        return points_norm.astype(np.float32).flatten()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        camera_intrinsics, _ = self.load_config(folder_id)
        camera_matrix = np.array(camera_intrinsics[0]['cam_K']).reshape(3, 3)

        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        img = self.load_image(img_path)

        translation, rotation, bbox, obj_id = self.load_6d_pose(folder_id, sample_id)

        # Project the 3D bounding box points
        # The caching logic here is good for avoiding re-projection
        cache_key = f"{folder_id}-{sample_id}"
        if cache_key not in self.points2d_cache:
            points_2d = self.get_3d_bbox_projection(obj_id, rotation, translation, camera_matrix)
            points_norm = self.normalize_points(points_2d, bbox)
            self.points2d_cache[cache_key] = points_norm
        else:
            points_norm = self.points2d_cache[cache_key]

        # Crop the image using the bounding box with padding
        # Note: img is already a tensor after self.load_image.
        # Converting it back to PIL Image here might be inefficient.
        # It's better to apply cropping and resizing on the PIL Image
        # BEFORE converting to tensor and normalizing.
        # However, for consistency with the original code flow,
        # I'll keep the ToPILImage conversion here but note the potential for optimization.
        img_pil = transforms.ToPILImage()(img) # Convert tensor back to PIL Image for cropping
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Add 20% padding
        pad_x = int(0.2 * (x_max - x_min))
        pad_y = int(0.2 * (y_max - y_min))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(img_pil.width, x_max + pad_x)
        y_max = min(img_pil.height, y_max + pad_y)

        cropped = img_pil.crop((x_min, y_min, x_max, y_max))
        cropped_resized = transforms.Resize((224, 224))(cropped)
        cropped_tensor = self.transform(cropped_resized) # Apply the dataset's transform (ToTensor, Normalize)

        # Load the original image without normalization for potential visualization later
        original_img_tensor = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

        return {
            "rgb": cropped_tensor,
            "points_2d": torch.tensor(points_norm),
            "obj_id": torch.tensor(obj_id - 1),  # Convert to 0-based index
            "original_img": original_img_tensor, # Use the correctly loaded original image
            "bbox": torch.tensor(bbox),
            "rotation": torch.tensor(rotation),
            "translation": torch.tensor(translation),
            "camera_matrix": torch.tensor(camera_matrix),
        }

# Data transformations
# These transforms are now applied inside the LinemodDataset's __getitem__
# for the cropped image, and separately for the full original image.
# The `transform` attribute of LinemodDataset is used for the cropped_tensor.
# The `load_normal_image` function uses `transforms.ToTensor()`.

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
def create_dataloaders(data_dir, batch_size=16):
    # Pass the appropriate transforms to the dataset constructors
    train_dataset = LinemodDataset(data_dir, split='train')
    val_dataset = LinemodDataset(data_dir, split='test')

    # It's good practice to set num_workers based on CPU cores, but be mindful of memory.
    # prefetch_factor can help with data loading efficiency.
    # num_workers=4 is a reasonable default for Colab. multiprocessing.cpu_count() might be too high
    # if it leads to excessive memory usage or context switching overhead.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

    return train_loader, val_loader


# --- train_model function ---
def train_model(batch_size = 128, num_epochs=10):
    # Configuration
    dataset_root = "Linemod_preprocessed/" # Ensure this path is correct
    learning_rate = 0.001
    checkpoint_dir = "checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    # Note: The transforms defined outside are not directly used by LinemodDataset's __init__
    # The LinemodDataset has its own internal `self.transform`
    train_dataset = LinemodDataset(dataset_root, split='train')
    val_dataset = LinemodDataset(dataset_root, split='test')

    # It's good practice to set num_workers based on CPU cores, but be mindful of memory.
    # prefetch_factor can help with data loading efficiency.
    # Changed num_workers to 4 as multiprocessing.cpu_count() can be too high for Colab
    # leading to memory issues.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

    # Model and optimizer
    # Renamed PoseModel to BB8Model to match the class definition provided
    model = BB8Model(num_objects=15).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error Loss for keypoint regression

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        train_loss = 0.0

        # Training loop with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images = batch['rgb'].cuda()
            targets = batch['points_2d'].cuda()

            pred_points, _ = model(images) # Forward pass: predict 2D points and ignore symmetry output

            # This part seems redundant if pred_points already has the correct shape (batch_size, 16)
            # and is directly used for loss. Assuming pred_points is already (batch_size, 16).
            # If pred_points is (batch_size, N, 16) where N is some intermediate dimension,
            # then this selection might be needed. However, based on BB8Model.bbox_head,
            # it should directly output (batch_size, 16).
            # Keeping it as is to avoid breaking existing logic, but it's worth reviewing.
            batch_size_actual = images.size(0)
            selected_preds = torch.zeros(batch_size_actual, 16).cuda()
            for i in range(batch_size_actual):
                selected_preds[i] = pred_points[i]

            loss = criterion(selected_preds, targets) # Calculate loss between predicted and target points
            optimizer.zero_grad() # Zero the gradients before backpropagation
            loss.backward() # Perform backpropagation
            optimizer.step() # Update model parameters
            train_loss += loss.item() # Accumulate training loss

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                images = batch['rgb'].cuda()
                targets = batch['points_2d'].cuda()

                pred_points, _ = model(images) # Forward pass

                # Same redundancy note as above for selected_preds
                batch_size_actual = images.size(0)
                selected_preds = torch.zeros(batch_size_actual, 16).cuda()
                for i in range(batch_size_actual):
                    selected_preds[i] = pred_points[i]

                loss = criterion(selected_preds, targets) # Calculate validation loss
                val_loss += loss.item() # Accumulate validation loss

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

# --- Main execution block (for demonstration) ---
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = 128
# Step 2: Train the model
print(f"Starting model training...")
trained_model = train_model(batch_size=batch_size, num_epochs=5)
print(f"Training complete.")

# Save only the model weights (state_dict) after training
torch.save(trained_model.state_dict(), os.path.join("checkpoints", "final_model.pth"))
print("✅ Final model weights saved to checkpoints/final_model.pth")