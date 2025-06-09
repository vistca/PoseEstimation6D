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


def load_model_info(dataset_root):
        obj_path = os.path.join(dataset_root, 'models', "models_info.yml")
        with open(obj_path, 'r') as f:
            models_info = yaml.load(f, Loader=yaml.FullLoader)

        return models_info

def transform_data(image, bounding_box, padding, dims):
    img = transforms.ToPILImage()(image)
    img = rgb_crop_img(img, bounding_box, padding)
    img = img.resize(dims)
    img = np.array(img).astype(np.float32)
    return torch.from_numpy(img)


class Tester():

    def __init__(self, boxModel, poseModel, dataset_root, padding=0.2):
        
        self.boxModel = boxModel
        self.poseModel = poseModel
        self.dims = self.poseModel.get_dimensions()
        self.padding = padding
        
        self.model_info = load_model_info(dataset_root)
        self.ply_objs = get_ply_files()

    def validate(self, val_loader, device, type_ds="Test"):
        # Validation phase
        self.boxModel.eval()
        self.poseModel.eval()
        add_total = [0,0]
        add_objects = {}

        with torch.no_grad(): # Disable gradient calculation for validation
            desc = "Test"
            progress_bar = tqdm(val_loader, desc=desc, ncols=100)

            for batch_id, batch in enumerate(progress_bar):
                inputs = {}
                inputs["rgb"] = batch["rgb"].to(device)
                
                # Calculates the bbox and ids for each sample
                outputs = self.boxModel(inputs["rgb"])
                # Output: [{bbox1, label1}, {bbox2, label2}]
                
                # TODO: Check that this actually produces the correct thing
                # what we want to do is to convert the shape to the same as we would 
                # otherwise expect the bbox and id to be
                # TODO: Might not need to send to device
                bboxes = np.empty((batch_size, 4))
                ids = np.empty((batch_size))
                for i in range(len(outputs)):
                    ids[i] = outputs[i]["labels"]
                bboxes = torch.from_numpy(bboxes).to(device)
                ids = torch.from_numpy(ids).to(device)
                

                # Pass this to the second model responsible for pose est
                
                nr_datapoints = bboxes.shape[0]

                images = np.empty((batch_size, 3, self.dims[0], self.dims[1]))
                depths = np.empty((batch_size, 1, self.dims[0], self.dims[0]))
                for i in range(nr_datapoints):
                    img = inputs["rgb"][i]
                    depth = batch["depth"][i]
                    bbox = bboxes[i]
                    # We have to permutate since PIL changes the shape from (C, H, W) -> (H, W, C)
                    img = transform_data(img, bbox, self.padding, self.dims).permute(2, 0, 1)
                    depth = transform_data(depth, bbox, self.padding, self.dims).unsqueeze(0)
                    images[i] = img
                    depths[i] = depth

                images = torch.from_numpy(images).float().to(device)
                depths = torch.from_numpy(depths).float().to(device)
                inputs = {"rgb" : images, "depth" : depths}
                

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

                    model_points = self.ply_objs[int(ids[i].item()+1)]

                    add = compute_ADD(model_points, gt_R, gt_t, pred_R, pred_t)

                    add_obj = add_objects.get(str(int(ids[i].item()+1)))
                    if not add_obj:
                        new_count = 1
                        new_val = add
                    else:
                        new_count = add_obj[0] + 1
                        new_val = add_obj[1] + add
                    add_objects[str(int(ids[i].item()+1))] = [new_count, new_val]
                    add_total = [add_total[0] + 1, add_total[1] + add]

                progress_bar.set_postfix(total="Placeholder")

        avg_add_total = add_total[1] / add_total[0]
        
        for k,v in add_objects.items():
            avg_add_obj = v[1] / v[0]
            print(f"Obj: {k}, Avg ADD: {avg_add_obj}, num obj: {v[0]}")
        print(f"Total average ADD: {avg_add_total}")
    

if __name__ == "__main__":
    batch_size = 2

    box_model_name = "transform"
    box_model_load_name = "Transform_aug"
    box_trainable_backbone = 5


    comb_model_name = "bb8_1"
    extension_model_load_name = "extension_test_15_92"


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
    print("Bbox model loaded")


    extension_model = CombinedModel(device, comb_model_name)
    load_path = f"extension/checkpoints/{extension_model_load_name}.pt"
    extension_model.load_state_dict(torch.load(load_path, weights_only=True, map_location=device.type))
    print("Extension models loaded")


    test_dataset = CombinedDataset(dataset_root, split_percentage, extension_model.get_dimensions(), split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    test_cls = Tester(bbox_model, extension_model, dataset_root)
    test_cls.validate(test_loader, device)