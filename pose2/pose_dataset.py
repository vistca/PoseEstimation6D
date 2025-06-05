import os
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
import torch
import yaml
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# --- LinemodDataset Class Definition ---
# This class is included here to make the code self-contained,
# as it was defined in a previous turn and is used by the training function.
class LinemodDataset(Dataset):
    def __init__(self, dataset_root, split_ratio, dimensions, split='train', seed=42):
        self.dataset_root = dataset_root
        self.split = split
        self.seed = seed
        self.split_ratio = split_ratio
        self.dimensions = dimensions
        self.samples = self.get_all_samples()

        print(f"Loading the {split} data ...")

        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}.")
        
        # Split into training and test sets
        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=(1-self.split_ratio['test_%']), random_state=self.seed
        )
        
        adjusted_train_perc = self.split_ratio['train_%'] / (self.split_ratio['train_%'] + self.split_ratio['val_%'])

        self.train_samples, self.val_samples = train_test_split(
            self.train_samples, train_size=adjusted_train_perc, random_state=42
        )

        # Select the appropriate split
        if self.split == "train":
            self.samples = self.train_samples
        elif self.split == "val":
            self.samples = self.val_samples
        else:
            self.samples = self.test_samples


        # This transform is applied to the *cropped* image in __getitem__
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            #transforms_v2.GaussianNoise(mean=0., sigma=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # This transform is for the *original* image if needed unnormalized
        self.to_tensor_only = transforms.ToTensor()

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
        for folder_id in tqdm(range(1, 16)):
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

    # Removed load_image and load_normal_image as they are now integrated into __getitem__
    # for more efficient processing.

    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.gt_cache.get(folder_id, {})
        # Use sample_id directly as an integer, as keys in gt.yml are typically integers

        if sample_id not in pose_data:
            print(f"id: {sample_id},{pose_data}")
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        poses = pose_data[sample_id]
        
        pose = None # pose[0]
        for temp_pose in poses:
            if temp_pose['obj_id'] == int(folder_id):
                pose = temp_pose
        
        [0] # <--- point of concern
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
    def get_3d_bbox_points(self, obj_id):
        """Obtains the 3D bounding box points for an object from dataset's models_info"""
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

        return points_3d


    def get_3d_bbox_projection(self, obj_id, rotation, translation, camera_matrix):
        """Projects the 3D bounding box points into the 2D image"""
        # Call the helper method within the dataset itself
        points_3d = self.get_3d_bbox_points(obj_id)
        points_2d, _ = cv2.projectPoints(points_3d, rotation, translation,
                                        camera_matrix, None)
        return points_2d.squeeze()

    def normalize_points(self, points_2d, bbox):
        """Normalizes the 2D points with respect to the bounding box"""
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Avoid division by zero if width or height is zero
        if width == 0 or height == 0:
            # Handle this case, e.g., return zeros or raise an error
            return np.zeros_like(points_2d).astype(np.float32).flatten()

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

        # Load original image as PIL Image once
        original_img_pil = Image.open(img_path).convert("RGB")

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

        # Crop the image using the bounding box with padding directly on PIL Image
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Add 20% padding
        pad_x = int(0.2 * (x_max - x_min))
        pad_y = int(0.2 * (y_max - y_min))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(original_img_pil.width, x_max + pad_x)
        y_max = min(original_img_pil.height, y_max + pad_y)

        cropped_pil = original_img_pil.crop((x_min, y_min, x_max, y_max))
        cropped_resized_pil = transforms.Resize(self.dimensions)(cropped_pil)

        # Apply the dataset's transform (ToTensor, Normalize) to the cropped image
        cropped_tensor = self.transform(cropped_resized_pil)

        # For 'original_img' output, apply ToTensor to the uncropped, unnormalized PIL image
        original_img_tensor = self.to_tensor_only(original_img_pil)

        points_3d = self.get_3d_bbox_points(obj_id)

        return {
            "rgb": cropped_tensor,
            "points_2d": torch.tensor(points_norm),
            "points_3d": torch.tensor(points_3d),
            "obj_id": torch.tensor(obj_id - 1),  # Convert to 0-based index
            "original_img": original_img_tensor,
            "bbox": torch.tensor(bbox),
            "rotation": torch.tensor(rotation),
            "translation": torch.tensor(translation),
            "camera_matrix": torch.tensor(camera_matrix),
        }
    
    # ##
# # Data transformations (These are now primarily used by the LinemodDataset's internal `self.transform`)
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Create datasets and dataloaders
# def create_dataloaders(data_dir, batch_size=16):
#     # The LinemodDataset's __init__ already sets up its internal transforms.
#     # The `train_transform` and `val_transform` defined globally are not directly passed here,
#     # but the dataset's internal logic handles the transformations.
#     train_dataset = LinemodDataset(data_dir, split='train')
#     val_dataset = LinemodDataset(data_dir, split='test')

#     # num_workers=4 is a reasonable default for Colab. multiprocessing.cpu_count() might be too high
#     # if it leads to excessive memory usage or context switching overhead.
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

#     return train_loader, val_loader
# ##