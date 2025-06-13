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
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# --- LinemodDataset Class Definition ---
# This class is included here to make the code self-contained,
# as it was defined in a previous turn and is used by the training function.
class PoseEstDataset(Dataset):
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


        # Define image transformations
        self.train_transform = transforms.Compose([
            transforms.Resize(self.dimensions),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize(self.dimensions),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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
        
        #[0] # <--- point of concern
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
        diameter = info['diameter']

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

        return points_3d, diameter


    def get_3d_bbox_projection(self, obj_id, rotation, translation, camera_matrix):
        """Projects the 3D bounding box points into the 2D image"""
        # Call the helper method within the dataset itself
        points_3d, _ = self.get_3d_bbox_points(obj_id)
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
    
    def load_image(self, img_path, bbox, padding=0.1):
        """Load an RGB image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        img = self.rgb_crop_img(img, bbox, padding)
        #img = img.resize(self.dimensions)
        
        #If we want to visialize the img crop
        #draw = ImageDraw.Draw(img)
        #font = ImageFont.load_default()
        #text = img_path
        #position = (10, 10)
        #draw.text(position, text, fill=(128, 0, 128), font=font)
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()

        if self.split == "train":
            return self.train_transform(img)
        
        return self.val_test_transform(img)


    def load_depth(self, depth_path, bbox, padding=0.1):
        """Load a depth image and convert to tensor."""
        
        x_min, y_min, x_max, y_max = map(int, bbox)
        m = padding

        pad_x = int(m * (x_max - x_min))
        pad_y = int(m * (y_max - y_min))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(640, x_max + pad_x)
        y_max = min(480, y_max + pad_y)

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        cropped = depth_raw[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, self.dimensions)
        depth = resized.astype(np.float32) / 1000.0
        
        # If we want to visualize the depth crop
        # plt.imshow(depth)
        # plt.axis("off")
        # plt.show()

        return torch.tensor(depth)

    
    def rgb_crop_img(self, rgb_img, b, m): # b is the bounding box for the image and m is the wanted margin
        # b = [x_left, y_top, x_width, y_height]
        x_min, y_min, x_max, y_max = map(int, b)

        pad_x = int(m * (x_max - x_min))
        pad_y = int(m * (y_max - y_min))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(rgb_img.width, x_max + pad_x)
        y_max = min(rgb_img.height, y_max + pad_y)

        return rgb_img.crop((x_min , y_min, x_max, y_max))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        camera_intrinsics, _ = self.load_config(folder_id)
        camera_matrix = np.array(camera_intrinsics[0]['cam_K']).reshape(3, 3)

        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"depth/{sample_id:04d}.png")

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


    
        cropped_tensor = self.load_image(img_path, bbox, 0.2)
        depth_data = self.load_depth(depth_path, bbox, 0.2)
        depth_data = depth_data.unsqueeze(0)

        points_3d, diameter = self.get_3d_bbox_points(obj_id)

        return {
            "rgb": cropped_tensor,
            "depth": depth_data,
            "points_2d": torch.tensor(points_norm),
            "points_3d": torch.tensor(points_3d),
            "obj_id": torch.tensor(obj_id - 1),  # Convert to 0-based index
            "bbox": torch.tensor(bbox),
            "rotation": torch.tensor(rotation),
            "translation": torch.tensor(translation),
            "camera_matrix": torch.tensor(camera_matrix),
            "diameter": torch.tensor(diameter),
        }