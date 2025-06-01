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
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class PoseDataset(Dataset):
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

        # Get list of all samples (folder_id, sample_id)
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
            self.train_samples, train_size=adjusted_train_perc, random_state=42
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.load_sample_confs()

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
    
    def rgb_crop_img(self, rgb_img, b, m): # b is the bounding box for the image and m is the wanted margin
        # b = [x_left, y_top, x_width, y_height]
        x_min = b[0] - m * b[2]
        x_max = b[0] + (m + 1) * b[2]

        y_min = b[1] - m * b[3]
        y_max = b[1] + (m + 1) * b[3]
        return rgb_img.crop((x_min , y_min, x_max, y_max))

    def load_image(self, img_path, bbox):
        """Load an RGB image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        img = self.rgb_crop_img(img, bbox, 0.1)

        img = img.resize(self.dimensions)
        # draw = ImageDraw.Draw(img)

        # # 3. Pick a font (here we use the default; you can also load a TTF)
        # font = ImageFont.load_default()

        # # 4. Define the text and its position (10px from top and left)
        # text = img_path
        # position = (10, 10)  # (x, y) in pixels

        # # 5. Draw the text in purple (R=128, G=0, B=128)
        # draw.text(position, text, fill=(128, 0, 128), font=font)

        # # 6. Display with Matplotlib
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()

        if self.split == "train":
            return self.train_transform(img)
        
        return self.val_test_transform(img)
    
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

        #x_min, y_min, width, height = bbox
        #x_max = x_min + width
        #y_max = y_min + height
        #bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32) #x_min, y_min, x_max, y_max

        return translation, rotation, bbox, obj_id

    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)


    def __getitem__(self, idx):
        """Load a dataset sample."""
        og_folder_id, sample_id = self.samples[idx]
        folder_id = str(og_folder_id).zfill(2) 

        # Load the correct camera intrinsics and object info for this folder
        camera_intrinsics = self.cam_data[folder_id]
        #print(camera_intrinsics.keys())

        img_path = os.path.join(self.dataset_root, 'data', folder_id, f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', folder_id, f"depth/{sample_id:04d}.png")


        translation, rotation, bbox, obj_id = self.load_6d_pose(folder_id, sample_id)
        img = self.load_image(img_path, bbox)
        depth = self.load_depth(depth_path)
        point_cloud = self.load_point_cloud(depth.numpy(), camera_intrinsics)
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)



        # TODO: Look at tensor creation "sourceTensor.clone().detach().requires_grad_(True)" instead of torch.tensor()
        return {
            "rgb": img,
            # "depth": depth.clone().detach(), #torch.tensor(depth, dtype=torch.float32),
            # "point_cloud": point_cloud,
            "camera_intrinsics": camera_intrinsics['0']['cam_K'],
            "objects_info": self.objects_info[og_folder_id],
            "translation": torch.tensor(translation),
            "rotation": torch.tensor(rotation),
            "bbox": torch.tensor(bbox),
            "obj_id": torch.tensor(obj_id)
        }


    