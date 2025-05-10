import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split

class RgbDataset(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        """
        Args:
            dataset_root (str): Path to the dataset directory.
            split (str): 'train' or 'test'.
            train_ratio (float): Percentage of data used for training (default 80%).
            seed (int): Random seed for reproducibility.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        # Get list of all samples (folder_id, sample_id)
        self.samples = self.get_all_samples()

        # Check if samples were found
        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}. Check the dataset path and structure.")

        # Split into training and test sets
        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed
        )

        # Select the appropriate split
        self.samples = self.train_samples if split == 'train' else self.test_samples

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.pose_data = {}
        dirs = os.listdir(self.dataset_root + 'data')
        for dir in dirs:
            """Load the 6D pose (translation and rotation) for the object in this sample."""
            pose_file = os.path.join(self.dataset_root, 'data', f"{dir:02d}", "gt.yml")

            # Load the ground truth poses from the gt.yml file
            with open(pose_file, 'r') as f:
                pose_data = yaml.load(f, Loader=yaml.FullLoader)
            
            self.pose_data[dir] = pose_data


    def get_all_samples(self):
        """Retrieve the list of all available sample indices from all folders."""
        samples = []
        for folder_id in range(1, 16):  # Assuming folders are named 01 to 15
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "rgb")
            #print(folder_path)
            if os.path.exists(folder_path):
                sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
                samples.extend([(folder_id, sid) for sid in sample_ids])  # Store (folder_id, sample_id)
        return samples

    def load_image(self, img_path):
        """Load an RGB image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.pose_data[folder_id]

        # The pose data is a dictionary where each key corresponds to a frame with pose info
        # We assume sample_id corresponds to the key in pose_data
        if sample_id not in pose_data:
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        pose = pose_data[sample_id][0]  # There's only one pose per sample

        # Extract translation and rotation
        bbox = np.array(pose['obj_bb'], dtype=np.float32) #[4] ---> x_min, y_min, width, height
        obj_id = np.array(pose['obj_id'], dtype=np.float32) #[1] ---> label

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32) #x_min, y_min, x_max, y_max

        return bbox, obj_id

    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a dataset sample."""
        folder_id, sample_id = self.samples[idx]
        # Load the correct camera intrinsics and object info for this folder

        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")

        img = self.load_image(img_path)
        bbox, obj_id = self.load_6d_pose(folder_id, sample_id)

        # TODO: Look at tensor creation "sourceTensor.clone().detach().requires_grad_(True)" instead of torch.tensor()
        return {
            "rgb": img,
            "bbox": torch.tensor(bbox),
            "obj_id": torch.tensor(obj_id)

        }
