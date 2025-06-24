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
from pose2.models.model_creator import create_model
from bbox.models.fasterRCNN import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from PIL import Image, ImageDraw, ImageFont


def load_model_info(dataset_root):
        obj_path = os.path.join(dataset_root, 'models', "models_info.yml")
        with open(obj_path, 'r') as f:
            models_info = yaml.load(f, Loader=yaml.FullLoader)

        return models_info

def transform_data(image, bounding_box, padding, dims, is_img=False):
    if is_img:
        mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
        std  = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
        img_t = image * std + mean
        image = img_t.clamp(0.0, 1.0)

    img = transforms.ToPILImage()(image)
    img.show()
    img = rgb_crop_img(img, bounding_box, padding)
    img = img.resize(dims)
    img.show()
    img = np.array(img).astype(np.float32)
    return torch.from_numpy(img).permute(2, 0, 1)


class Tester():

    def __init__(self, boxModel, poseModel, dataset_root, run_extension, dims, padding=0.2):
        
        self.boxModel = boxModel
        self.poseModel = poseModel
        self.dims = dims
        self.padding = padding
        self.run_extension = run_extension
        
        self.model_info = load_model_info(dataset_root)
        self.ply_objs = get_ply_files()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def validate(self, val_loader, device, type_ds="Test"):
        # Validation phase
        self.boxModel.eval()
        self.poseModel.eval()
        add_total = [0,0]
        add_objects = {}
        more_than_one_count = 0
        missmatch_obj = 0
        dia_10_objects = {}
        cm_2_objects = {}
        below_2cm = [0,0]
        below_dia = [0,0]

        with torch.no_grad(): # Disable gradient calculation for validation
            desc = "Test"
            progress_bar = tqdm(val_loader, desc=desc, ncols=100)

            for batch_id, batch in enumerate(progress_bar):
                inputs = {}
                inputs["rgb"] = batch["rgb"].to(device)
                inputs["obj_id"] = batch["obj_id"].to(device)
                diameters = batch["diameter"]
                outputs = self.boxModel(inputs["rgb"])
                nr_datapoints = len(outputs)

                bboxes = np.empty((nr_datapoints, 4))
                ids = np.empty((nr_datapoints))
                for i in range(len(outputs)):
                    #print(outputs[i]["boxes"].cpu().shape)
                    bbox = outputs[i]["boxes"]
                    obj_id = outputs[i]["labels"]
                    if len(obj_id) > 1:
                        more_than_one_count += 1
                        index = torch.argmax(obj_id)
                        obj_id = outputs[i]['labels'][index]
                        bbox = outputs[i]['boxes'][index]

                    tmp = inputs["obj_id"][i].item()
                    print(tmp)
                    print(obj_id.item())
                    if int(obj_id.item()) != int(tmp):
                      missmatch_obj += 1

                    bboxes[i] = bbox.cpu()
                    ids[i] = obj_id.cpu()
                bboxes = torch.from_numpy(bboxes).to(device)
                ids = torch.from_numpy(ids).to(device)
                

                images = np.empty((nr_datapoints, 3, self.dims[0], self.dims[1]))
                depths = np.empty((nr_datapoints, 1, self.dims[0], self.dims[1]))
                for i in range(nr_datapoints):
                    bounding_box = bboxes[i]
                    img_path = batch["img_path"][i]
                    img = Image.open(img_path) 
                    img = rgb_crop_img(img, bounding_box, self.padding)
                    img = img.resize(self.dims)
                    img = self.transform(img)

                    depth_path = batch["depth_path"][i]
                    depth = Image.open(depth_path)
                    depth = rgb_crop_img(depth, bounding_box, self.padding)
                    depth = depth.resize(self.dims)
                    depth = np.array(depth).astype(np.float32)
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    images[i] = img
                    depths[i] = depth

                images = torch.from_numpy(images).float().to(device)
                depths = torch.from_numpy(depths).float().to(device)
                if run_extension:
                    inputs = {"rgb" : images, "depth" : depths}
                else:
                    inputs = {"rgb": images}
                

                # Do a forward pass to the second model
                pred_points = self.poseModel(inputs)

            
                pred_points = rescale_pred(pred_points, bboxes, nr_datapoints)
                
                model_points = np.empty((nr_datapoints, 8, 3))
                for i in range(nr_datapoints):
                    model_point_3d = get_3d_bbox_points(self.model_info, int(ids[i].item()))#batch["points_3d"]
                    model_points[i] = model_point_3d 
                model_points = torch.from_numpy(model_points)

            
                reconstruction_3d = reconstruct_3d_points_from_pred(pred_points, model_points, nr_datapoints)


                gts_t = batch["translation"]
                gts_R = batch["rotation"]
                for i in range(nr_datapoints):

                    pred_t = reconstruction_3d[i, :3]
                    pred_R = reconstruction_3d[i, 3:].reshape((3,3))


                    gt_t = gts_t[i]
                    gt_R = gts_R[i].reshape((3,3))

                    model_points = self.ply_objs[int(ids[i].item())]

                    add = compute_ADD(model_points, gt_R, gt_t, pred_R, pred_t)

                    add_total = [add_total[0] + add, add_total[1] + 1]

                    add_obj = add_objects.get(str(int(ids[i].item())))
                    if not add_obj:
                        new_count = 1
                        new_val = add
                    else:
                        new_count = add_obj[0] + 1
                        new_val = add_obj[1] + add
                    add_objects[str(int(ids[i].item()))] = [new_count, new_val]
                    
                    
                    below_2cm[1] = below_2cm[1] + 1
                    if add < 20:
                        below_2cm[0] = below_2cm[0] + 1

                    cm_2_object = cm_2_objects.get(str(int(ids[i].item())))
                    diameter = diameters[i]
                    new_low = 1 if add < 20 else 0
                    if not cm_2_object:
                        new_count = 1
                        new_val = new_low
                    else:
                        new_count = cm_2_object[0] + 1
                        new_val = cm_2_object[1] + new_low
                    cm_2_objects[str(int(ids[i].item()))] = [new_count, new_val]

                    dia_10_object = dia_10_objects.get(str(int(ids[i].item())))
                    diameter = diameters[i]
                    new_low = 1 if add < 0.1*diameter else 0
                    if not dia_10_object:
                        new_count = 1
                        new_val = new_low
                    else:
                        new_count = dia_10_object[0] + 1
                        new_val = dia_10_object[1] + new_low
                    dia_10_objects[str(int(ids[i].item()))] = [new_count, new_val]
                    below_dia = [below_dia[0] + new_low, below_dia[1] + 1]

                progress_bar.set_postfix(total="Placeholder")


        avg_add_total = add_total[0] / add_total[1]
        percentage_below_2cm = 100*below_2cm[0]/below_2cm[1]
        percentage_below_10_dia = 100*below_dia[0]/below_dia[1]

        print("Multi-preds made by the model: ", more_than_one_count)
        print("Missmatch obj-preds made by the model: ", missmatch_obj)

        print("")
        
        for k,v in add_objects.items():
            avg_add_obj = v[1] / v[0]
            print(f"Obj: {k}, Avg ADD: {avg_add_obj}, num obj: {v[0]}")
        print(f"Total average ADD: {avg_add_total}")

        print("")

        for k,v in dia_10_objects.items():
            perc_below_dia = 100*v[1] / v[0]
            print(f"Obj: {k}, Percentage below 10% of diameter: {perc_below_dia}%, num obj: {v[0]}")
        print(f"Total percentage below 10% of diameter: {percentage_below_10_dia}%")

        print("")

        for k,v in cm_2_objects.items():
            perc_below_2_cm = 100*v[1] / v[0]
            print(f"Obj: {k}, Percentage below 2cm: {perc_below_2_cm}%, num obj: {v[0]}")
        print(f"Percentage below 2cm : {percentage_below_2cm}%")
    

if __name__ == "__main__":
    batch_size = 16
    run_extension = False

    box_model_name = "transform"
    #box_model_load_name = "Transform_3tr_ep3"
    box_model_load_name = "Transform_aug"
    box_trainable_backbone = 3


    comb_model_name = "bb8_1"
    #extension_model_load_name = "extension_test_15_92"


    split_percentage = {
                        "train_%" : 0.7,
                        "test_%" : 0.15,
                        "val_%" : 0.15,
                        }

    dataset_root = dataset_root = "./datasets/Linemod_preprocessed/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    bbox_model = FasterRCNN(box_trainable_backbone, box_model_name)
    bbox_model = bbox_model.get_model()
    #fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=5)
    load_path = f"bbox/checkpoints/{box_model_load_name}.pt"
    bbox_model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
    bbox_model.to(device)
    print("Bbox model loaded")

    if run_extension:
        pose_model_load_name = "extension_test_11_31"
        model = CombinedModel(device, comb_model_name)
        load_path = f"extension/checkpoints/{pose_model_load_name}.pt"
        model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
        print("Extension models loaded")
        model.to(device)
        dims = model.get_dimensions()

    else:
        model_name = "bb8_1"
        pose_model_load_name = "pose_model"
        load_path = f"pose2/checkpoints/{pose_model_load_name}.pt"
        model = create_model(model_name) 
        model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
        print("RGB data pose model loaded")
        model.to(device)
        dims = model.get_dimension()

    test_dataset = CombinedDataset(dataset_root, split_percentage, dims, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_cls = Tester(bbox_model, model, dataset_root, run_extension, dims)
    test_cls.validate(test_loader, device)
