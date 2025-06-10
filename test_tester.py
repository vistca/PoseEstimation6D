from pose2.utils.models_points import get_ply_files
from pose2.utils.data_reconstruction import reconstruct_3d_points_from_pred, rescale_pred
from pose2.utils.add_calc import compute_ADD
from utils.get_3d_points import get_3d_bbox_points
from utils.crop_rescale_img import rgb_crop_img
import torch
from tqdm import tqdm
import os
import yaml
from torchvision import transforms
import numpy as np
from combined_dataset import CombinedDataset
from torch.utils.data import DataLoader
from extension.models.combined_model import CombinedModel
from bbox.models.fasterRCNN import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import matplotlib.pyplot as plt

def load_pose_data(dataset_root):
    pose_data = {}

    """
        This loads the data into memory instead of having to access it each time 
        items are fetched from __getitem__
    """
    dirs = os.listdir(dataset_root + '/data')
    for dir in tqdm(dirs):
        if os.path.isdir(dataset_root + "/data/" + dir):
            pose_file = os.path.join(dataset_root, 'data', dir, "gt.json")

            # Load the ground truth poses from the gt.yml file
            with open(pose_file, 'r') as f:
                pose_data[dir] = json.load(f)

    return pose_data

def calc_add(reconstruction_3d,obj, gts_t, gts_R):
    ply_objs = get_ply_files()
    pred_t = reconstruction_3d[0, :3]
    pred_R = reconstruction_3d[0, 3:].reshape((3,3))

    gt_t = gts_t
    gt_R = gts_R.reshape((3,3))

    model_points = ply_objs[obj]

    add = compute_ADD(model_points, gt_R, gt_t, pred_R, pred_t)
    print("This was the add: ", add)

def rec_3d_points_from_pred(preds: torch.Tensor, models_points_3d, nr_datapoints, flatten=False):

    camera_matrix = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    result = torch.empty(nr_datapoints, 12, dtype=torch.float32)

    for i in range(nr_datapoints):
        pred = preds[i]
        pred_points = pred.squeeze(0)

        model_points_2d = np.empty((8,2))
        
        nr_points = pred_points.size()[0] // 2


        for j in range(nr_points):
            point_x = pred_points[2*j].item() 
            point_y = pred_points[2*j+1].item() 
            model_points_2d[j] = point_x, point_y

        model_points_3d = models_points_3d[i]
        success, pred_rvec, pred_pos = cv2.solvePnP(
            objectPoints=model_points_3d.numpy(),
            imagePoints=model_points_2d,
            cameraMatrix=camera_matrix,
            distCoeffs=np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        pred_rot_matrix, _ = cv2.Rodrigues(pred_rvec)
        pred_t = torch.tensor(pred_pos, dtype=torch.float32).squeeze()
        pred_R = torch.tensor(pred_rot_matrix, dtype=torch.float32).flatten()

        result[i] = torch.cat((pred_t, pred_R), dim=0)

    return result

def rescale_pred(preds: torch.Tensor, bboxes: torch.Tensor, nr_datapoints):

    for i in range(nr_datapoints):
        pred = preds[i]
        pred_points = pred.squeeze(0)

        nr_points = pred_points.size()[0] // 2

        x_min = bboxes[i, 0]
        y_min = bboxes[i, 1]
        x_max = bboxes[i, 2] 
        y_max = bboxes[i, 3]        
        
        width = x_max - x_min
        height = y_max - y_min

        for j in range(nr_points):
            pred_points[2*j] = pred_points[2*j] * width + x_min
            pred_points[2*j+1] = pred_points[2*j+1] * height + y_min

        preds[i] = pred_points

    return preds


if __name__ == "__main__":
    dataset_root = "./datasets/Linemod_preprocessed/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    comb_model_name = "bb8_1"
    #extension_model_load_name = "extension_test_11_31"
    extension_model_load_name = "extension_test_11_31"
    #extension_model_load_name = "extension_test_2_1"
    
    obj_id = 5
    folder_id = "05"
    sample_id = "313"
    img_path = "./datasets/Linemod_preprocessed/data/05/rgb/0313.png"
    depth_path = "./datasets/Linemod_preprocessed/data/05/depth/0313.png"
    
    
    bbox = [257.4022,  97.1515, 338.0164, 242.8628]
    #bbox = [258.,  99., 338., 244.]
    dims = (224, 224)

    pose_data = load_pose_data(dataset_root)


    pose_data = pose_data[folder_id]
    sample_id = str(sample_id)
    # The pose data is a dictionary where each key corresponds to a frame with pose info
    # We assume sample_id corresponds to the key in pose_data
    if sample_id not in pose_data.keys():
        raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

    
    poses = pose_data[sample_id]
    
    pose = None # pose[0]
    for temp_pose in poses:
        if temp_pose['obj_id'] == int(folder_id):
            pose = temp_pose

    translation = np.array(pose['cam_t_m2c'], dtype=np.float32)  # [3] ---> (x,y,z)
    rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)  # [3x3] ---> rotation matrix

    gts_t = torch.tensor(translation)
    gts_R = torch.tensor(rotation)
    
    models_info = None
    obj_path = os.path.join(dataset_root, 'models', "models_info.yml")
    with open(obj_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.FullLoader)



    val_test_transform = transforms.Compose([
            transforms.Resize(dims),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    
    img = Image.open(img_path).convert("RGB")
    img = rgb_crop_img(img, bbox, 0.1)
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    img = val_test_transform(img).unsqueeze(0)
    #plt.show(img)

    depth = Image.open(depth_path)
    depth = rgb_crop_img(depth, bbox, 0.1)
    depth = depth.resize(dims)
    # plt.imshow(depth)
    # plt.axis("off")
    # plt.show()
    depth = np.array(depth).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)


    dataset_root = dataset_root = "./datasets/Linemod_preprocessed/"
    extension_model = CombinedModel(device, comb_model_name)
    load_path = f"extension/checkpoints/{extension_model_load_name}.pt"
    extension_model.load_state_dict(torch.load(load_path, map_location=device.type))
    print("Extension models loaded")
    extension_model.eval()
    
    inputs = {"rgb" : img, "depth" : depth}

    pred_points = extension_model(inputs)
    bboxes = torch.from_numpy(np.array(bbox)).unsqueeze(0)
    pred_points = rescale_pred(pred_points, bboxes, 1)

    model_point_3d = get_3d_bbox_points(models_info, obj_id)#batch["points_3d"]
    model_point_3d = torch.from_numpy(model_point_3d).unsqueeze(0)
    reconstruction_3d = rec_3d_points_from_pred(pred_points, model_point_3d, 1)
    pred_t = reconstruction_3d[0, :3]
    pred_R = reconstruction_3d[0, 3:].reshape((3,3))
    
    print(pred_t)
    print(gts_t)
    
    print(pred_R)
    print(gts_R)
    calc_add(reconstruction_3d, obj_id, gts_t, gts_R)

    # Point pred vs GT (Gt second)
    #0.3994, 0.6850, 0.4030, 0.7621, 0.5324, 0.7345, 0.5614, 0.7192, 0.3581, 0.0368, 0.3811, 0.1069, 0.5011, 0.1229, 0.4939, 0.0832
    #0.3963, 0.6848, 0.4011, 0.7603, 0.5326, 0.7343, 0.5616, 0.7197, 0.3577,  0.0370, 0.3807, 0.1064, 0.5006, 0.1216, 0.4955, 0.0839
    
    # Translation + Rotation, pred vs GT (GT second)
    #1.4105e+05,  1.0473e+05, -2.4858e+05,  9.9848e-01,  5.4325e-02, 8.9688e-03, -2.7136e-02,  3.4379e-01,  9.3865e-01,  4.7909e-02, -9.3747e-01,  3.4474e-01
    #1.4106e+05,  1.0473e+05, -2.4859e+05,  9.9881e-01,  4.8395e-02, 6.2389e-03, -2.2478e-02,  3.4285e-01,  9.3912e-01,  4.3309e-02, -9.3814e-01,  3.4353e-01