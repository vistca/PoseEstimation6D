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
import json
import open3d as o3d

class CombinedDataset(Dataset):
    def __init__(self, dataset_root, split_ratio, dimensions, split='train', seed=42):
        """
        Args:
            dataset_root (str): Path to the dataset directory.
            split (str): 'train' or 'test'.
            train_ratio (float): Percentage of data used for training (default 80%).
            seed (int): Random seed for reproducibility.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.dimensions = dimensions
        self.samples = self.get_all_samples()

        # Check if samples were found
        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}. Check the dataset path and structure.")

        # Split into training and test sets
        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=(1-self.split_ratio['test_%']), random_state=self.seed
        )
        
        adjusted_train_perc = self.split_ratio['train_%'] / (self.split_ratio['train_%'] + self.split_ratio['val_%'])

        self.train_samples, self.val_samples = train_test_split(
            self.train_samples, train_size=adjusted_train_perc, random_state=self.seed
        )

        # Select the appropriate split
        if self.split == "train":
            self.samples = self.train_samples
        elif self.split == "val":
            self.samples = self.val_samples
        else:
            self.samples = self.test_samples

        print(f"Nr samples {len(self.samples)} for {self.split}")

        # Define image transformations
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.points2d_cache = {}
        self.load_sample_confs()

    def load_all_gt(self):
        """Preloads all GT data with caching"""
        for folder_id in range(1, 16):
            self.load_gt_for_folder(folder_id)

    def load_sample_confs(self):
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
        
    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.pose_data[folder_id]
        sample_id = str(sample_id)
        # The pose data is a dictionary where each key corresponds to a frame with pose info
        # We assume sample_id corresponds to the key in pose_data
        if sample_id not in pose_data.keys():
            print(pose_data.keys())
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        
        poses = pose_data[sample_id]
        
        pose = None # pose[0]
        for temp_pose in poses:
            if temp_pose['obj_id'] == int(folder_id):
                pose = temp_pose
        #pose = pose_data[sample_id][0]  # There's only one pose per sample

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
    
    def load_image(self, img_path):
        """Load an RGB image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        #img = self.rgb_crop_img(img, bbox, padding)
        #img = img.resize(self.dimensions)
        
        #If we want to visialize the img crop
        #draw = ImageDraw.Draw(img)
        #font = ImageFont.load_default()
        #text = img_path
        #position = (10, 10)
        #draw.text(position, text, fill=(128, 0, 128), font=font)
        #plt.imshow(img)
        #plt.axis("off")
        #plt.show()

        if self.split == "train":
            return self.train_transform(img), img
        
        return self.val_test_transform(img), img
    
    def load_depth(self, depth_path):
        """Load a depth image and convert to tensor."""
        img_depth = Image.open(depth_path)
        #depth = self.rgb_crop_img(depth, bbox, padding)
        
        # If we want to visualize the depth crop
        #plt.imshow(depth)
        #plt.axis("off")
        #plt.show()

        #depth = depth.resize(self.dimensions)
        depth = np.array(img_depth).astype(np.float32)
        depth = torch.from_numpy(depth)
        #print(depth)
        return depth, img_depth

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

    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
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
    

    def __getitem__(self, idx):
        """Load a dataset sample."""
        og_folder_id, sample_id = self.samples[idx]
        folder_id = str(og_folder_id).zfill(2) 

        #(sample_id)
        #print(folder_id)

        # Load the correct camera intrinsics and object info for this folder
        camera_intrinsics = self.cam_data[folder_id]
        camera_matrix = np.array(camera_intrinsics['0']['cam_K']).reshape(3, 3)
        #print(camera_intrinsics.keys())

        img_path = os.path.join(self.dataset_root, 'data', folder_id, f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', folder_id, f"depth/{sample_id:04d}.png")

        img, original_img = self.load_image(img_path)
        depth, original_depth = self.load_depth(depth_path)
        point_cloud = self.load_point_cloud(depth.numpy(), camera_intrinsics)
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
        translation, rotation, bbox, obj_id = self.load_6d_pose(folder_id, sample_id)

        # cache_key = f"{folder_id}-{sample_id}"
        # if cache_key not in self.points2d_cache:
        #     points_2d = self.get_3d_bbox_projection(obj_id, rotation, translation, camera_matrix)
        #     points_norm = self.normalize_points(points_2d, bbox)
        #     self.points2d_cache[cache_key] = points_norm
        # else:
        #     points_norm = self.points2d_cache[cache_key]

        #img = self.load_image(img_path)
        #depth_data = self.load_depth(depth_path, bbox, 0.2)
        depth = depth.unsqueeze(0)

        #points_3d = self.get_3d_bbox_points(obj_id)

        return {
            "rgb": img,
            "depth": depth.clone().detach(),
            #"points_2d": torch.tensor(points_norm),
            #"points_3d": torch.tensor(points_3d),
            #"original_img" : original_img,
            #"original_depth" : original_depth,
            "img_path" : img_path,
            "depth_path" : depth_path,
            "obj_id": torch.tensor(obj_id),
            "bbox": torch.tensor(bbox),
            "rotation": torch.tensor(rotation),
            "translation": torch.tensor(translation),
            #"point_cloud": point_cloud,
            #"camera_intrinsics": camera_intrinsics['0']['cam_K'],
            #"objects_info": self.objects_info[og_folder_id],
        }


    