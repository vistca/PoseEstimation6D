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

def model_predict(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        prediction = model(image.unsqueeze(0))  # Add batch dimension
    return prediction

def load_model(model_path, device):

    # model is hardcoded for now

    wrapper = FasterRCNN(3, "transform")
    model = wrapper.get_model()  # Get the actual nn.Module
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_bbox(img, device):
    model_path = "bbox/checkpoints/peach-sweep-1.pt " # Transform_aug .pt"  # Adjust according to checkpoint

    model = load_model(model_path, device)

    pred = model_predict(model, img, device)
    pred_bbox = pred[0]["boxes"].tolist()
    pred_obj_id = pred[0]["labels"].tolist()
    
    return pred_bbox, pred_obj_id

def get_model_3d_points(info):
    x_size = info['size_x']
    y_size = info['size_y']
    z_size = info['size_z']

    half_x = x_size / 2
    half_y = y_size / 2
    half_z = z_size / 2

    model_points_3d = np.array([
        [-half_x, -half_y, -half_z],
        [half_x, -half_y, -half_z],
        [half_x, half_y, -half_z],
        [-half_x, half_y, -half_z],
        [-half_x, -half_y, half_z],
        [half_x, -half_y, half_z],
        [half_x, half_y, half_z],
        [-half_x, half_y, half_z]
    ], dtype=np.float32)
    return model_points_3d

def draw_points_from_R_t(draw, edges, pred_t, pred_R, points_3d, color):
    rotated_points = (pred_R @ points_3d.T).T

    resulting_points = []

    for i in range(8):
        translated_points = rotated_points[i] + pred_t.T
        point = project_to_2d(translated_points.numpy())
        resulting_points.append(point)

    for edge in edges:
        x1, y1 = resulting_points[edge[0]]
        x2, y2 = resulting_points[edge[1]]
        draw.line(((x1, y1), (x2, y2)), fill=color, width=2)

def project_to_2d(coord):
    K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ]) # The camera matrix

    # The two dimensional coordinates derived through projection
    x = (K[0, 0] * coord[0] / coord[2]) + K[0, 2]
    y = (K[1, 1] * coord[1] / coord[2]) + K[1, 2]

    return (x, y)
    

if __name__ == "__main__":
    dataset_root = "./datasets/Linemod_preprocessed/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    comb_model_name = "bb8_1"
    #extension_model_load_name = "extension_test_11_31"
    extension_model_load_name = "extension_test_11_31"
    #extension_model_load_name = "extension_test_2_1"

    obj_id_gt = 8
    sample_id = 20

    folder_id = (2-len(str(obj_id_gt))) * "0" + str(obj_id_gt)
    sample_id_4_digit = "0" * (4 - len(str(sample_id))) + str(sample_id)
    img_path = f"./datasets/Linemod_preprocessed/data/{folder_id}/rgb/{sample_id_4_digit}.png"
    depth_path = f"./datasets/Linemod_preprocessed/data/{folder_id}/depth/{sample_id_4_digit}.png"
    

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

    resize_transform = transforms.Compose([
        transforms.Resize(dims)
    ])

    val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    
    raw_img = Image.open(img_path).convert("RGB")
    
    img = val_test_transform(raw_img)
    

    pred_bbox, pred_obj_id = predict_bbox(img, device)

    pred_bbox = pred_bbox[0]
    pred_obj_id = pred_obj_id[0]

    gt_bbox = pose['obj_bb']
    gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]]
    gt_id = pose['obj_id']


    print("predicted bbox and id:")
    print(pred_bbox)
    print(pred_obj_id)

    print("ground truth bbox and id")
    print(gt_bbox)
    print(gt_id)


    
    crop = rgb_crop_img(raw_img, pred_bbox, 0.2)
    crop = resize_transform(crop)
    crop = val_test_transform(crop)
    crop = crop.unsqueeze(0)

    depth = Image.open(depth_path)
    depth = rgb_crop_img(depth, pred_bbox, 0.2)
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
    
    inputs = {"rgb" : crop, "depth" : depth}

    pred_points = extension_model(inputs)
    bboxes = torch.from_numpy(np.array(pred_bbox)).unsqueeze(0)
    pred_points = rescale_pred(pred_points, bboxes, 1)

    model_point_3d = get_3d_bbox_points(models_info, obj_id_gt)#batch["points_3d"]
    model_point_3d = torch.from_numpy(model_point_3d).unsqueeze(0)
    reconstruction_3d = rec_3d_points_from_pred(pred_points, model_point_3d, 1)
    pred_t = reconstruction_3d[0, :3]
    pred_R = reconstruction_3d[0, 3:].reshape((3,3))
    
    print(pred_t)
    print(gts_t)
    
    print(pred_R)
    print(gts_R)

    pred_model_info = models_info[pred_obj_id]
    gt_model_info = models_info[gt_id]

    pred_model_3d_points = get_model_3d_points(pred_model_info)
    gt_model_3d_points = get_model_3d_points(gt_model_info)
    
    draw = ImageDraw.Draw(raw_img)
    edges = [(0,1), (0,3), (1,2), (2,3), (4,5), (4,7), (5,6), (6,7), (0,4), (1,5), (2,6), (3,7)]

    draw_points_from_R_t(draw, edges, gts_t, gts_R, gt_model_3d_points, "#00D9FF")
    draw_points_from_R_t(draw, edges, pred_t, pred_R, pred_model_3d_points, "#FF7B00")


    raw_img.show()
