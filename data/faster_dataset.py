import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
import json
import open3d as o3d
from tqdm import tqdm

class FasterDataset(Dataset):
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
        self.cam_data = {}

        """
            This loads the data into memory instead of having to access it each time 
            items are fetched from __getitem__
        """
        print(f"Initializing {self.split} dataset")
        dirs = os.listdir(self.dataset_root + '/data')
        for dir in tqdm(dirs):
            if os.path.isdir(self.dataset_root + "/data/" + dir):
               
                
                pose_file = os.path.join(self.dataset_root, 'data', dir, "gt.json")
                cam_file = os.path.join(self.dataset_root, 'data', dir, "info.json")
                objects_info_path = os.path.join(self.dataset_root, 'models', f"models_info.yml")

                # Load the ground truth poses from the gt.yml file
                with open(pose_file, 'r') as f:
                    self.pose_data[dir] = json.load(f)

                with open(cam_file, 'r') as f:
                    self.cam_data[dir] = json.load(f)

                with open(objects_info_path, 'r') as f:
                    self.objects_info = yaml.load(f, Loader=yaml.FullLoader)
        
    """
    def load_config(self, folder_id):
        Load YAML configuration files for camera intrinsics and object info for a specific folder.
        camera_intrinsics_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", 'info.yml')
        objects_info_path = os.path.join(self.dataset_root, 'models', f"models_info.yml")

        with open(camera_intrinsics_path, 'r') as f:
            camera_intrinsics = yaml.load(f, Loader=yaml.FullLoader)

        with open(objects_info_path, 'r') as f:
            objects_info = yaml.load(f, Loader=yaml.FullLoader)

        return camera_intrinsics, objects_info
    """

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
    
    def load_depth(self, depth_path):
        """Load a depth image and convert to tensor."""
        depth = np.array(Image.open(depth_path))
        return torch.tensor(depth, dtype=torch.float32)
    
    def load_point_cloud(self, depth, intrinsics):
        """Convert depth image to point cloud using Open3D."""
        intrinsics = intrinsics['0']['cam_K']
        h, w = depth.shape
        fx, fy, cx, cy = intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]

        # Generate 3D points
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
        z = depth / 1000.0  # Convert to meters
        x = (xmap - cx) * z / fx
        y = (ymap - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        return point_cloud


    """def load_6d_pose(self, folder_id, sample_id):
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
        """
    

    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.pose_data[folder_id]
        sample_id = str(sample_id)
        # The pose data is a dictionary where each key corresponds to a frame with pose info
        # We assume sample_id corresponds to the key in pose_data
        if sample_id not in pose_data.keys():
            print(pose_data.keys())
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        pose = pose_data[sample_id][0]  # There's only one pose per sample

        # Extract translation and rotation
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32)  # [3] ---> (x,y,z)
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)  # [3x3] ---> rotation matrix
        bbox = np.array(pose['obj_bb'], dtype=np.float32) #[4] ---> x_min, y_min, width, height
        obj_id = np.array(pose['obj_id'], dtype=np.float32) #[1] ---> label

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32) #x_min, y_min, x_max, y_max

        return translation, rotation, bbox, obj_id


    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)

    """
    def __getitem__(self, idx):
        Load a dataset sample
        folder_id, sample_id = self.samples[idx]
        # Load the correct camera intrinsics and object info for this folder
        folder_id = str(folder_id).zfill(2) 
        img_path = os.path.join(self.dataset_root, 'data', folder_id, f"rgb/{sample_id:04d}.png")

        img = self.load_image(img_path)
        bbox, obj_id = self.load_6d_pose(folder_id, sample_id)

        # TODO: Look at tensor creation "sourceTensor.clone().detach().requires_grad_(True)" instead of torch.tensor()
        return {
            "rgb": img,
            "bbox": torch.tensor(bbox),
            "obj_id": torch.tensor(obj_id)

        }
    """
    
    def __getitem__(self, idx):
        """Load a dataset sample."""
        og_folder_id, sample_id = self.samples[idx]
        folder_id = str(og_folder_id).zfill(2) 

        # Load the correct camera intrinsics and object info for this folder
        camera_intrinsics = self.cam_data[folder_id]
        #print(camera_intrinsics.keys())

        img_path = os.path.join(self.dataset_root, 'data', folder_id, f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', folder_id, f"depth/{sample_id:04d}.png")

        img = self.load_image(img_path)
        depth = self.load_depth(depth_path)
        point_cloud = self.load_point_cloud(depth.numpy(), camera_intrinsics)
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
        translation, rotation, bbox, obj_id = self.load_6d_pose(folder_id, sample_id)

        # TODO: Look at tensor creation "sourceTensor.clone().detach().requires_grad_(True)" instead of torch.tensor()
        return {
            "rgb": img,
            "depth": depth.clone().detach(), #torch.tensor(depth, dtype=torch.float32),
            "point_cloud": point_cloud,
            "camera_intrinsics": camera_intrinsics['0']['cam_K'],
            "objects_info": self.objects_info[og_folder_id],
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "bbox": torch.tensor(bbox),
            "obj_id": torch.tensor(obj_id)

        }


    