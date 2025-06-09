import yaml
import torch
from utils.runtime_args import add_runtime_args
from .data.faster_dataset import FasterDataset
from .models.fasterRCNN import FasterRCNN

import matplotlib.pyplot as plt
import matplotlib.patches as patches

args = add_runtime_args()
dataset_root = args.data + "/Linemod_preprocessed"


# Load images to use as input
# Load gt data for image bounding boxes

def load_data(): 
    with open('config/global_runtime_config.yaml') as f:
        config_dict = yaml.safe_load(f)

    split_percentage = {
                        "train_%" : config_dict["train_%"],
                        "test_%" : config_dict["test_%"],
                        "val_%" : config_dict["val_%"],
                        }
    
    val_dataset = FasterDataset(dataset_root, split_percentage, split="val")
    return val_dataset


# Load model to be used for predictions

def load_model(model_path, device):

    # model is hardcoded for now

    wrapper = FasterRCNN(0, "transform")
    model = wrapper.get_model()  # Get the actual nn.Module
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def model_predict(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        prediction = model(image.unsqueeze(0))  # Add batch dimension
    return prediction

def unnormalize(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # img: torch.Tensor (C, H, W)
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

# Draw predicted bb output and gt bb

def visualize(image, pred_bbox, pred_label, score, gt_bbox):
    
    img_disp = unnormalize(image.clone()) #unnormalize image
    img_disp = img_disp.clamp(0, 1)
    plt.imshow(img_disp.permute(1, 2, 0).cpu().numpy())
    
    ax = plt.gca()
    # Draw predicted bbox in red
    ax.add_patch(patches.Rectangle(
        (pred_bbox[0], pred_bbox[1]), pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1],
        linewidth=2, edgecolor='r', facecolor='none', label='Predicted'
    ))
    # labels
    ax.text(pred_bbox[0], pred_bbox[1] - 30, f'label_pred: {pred_label}', color='red', fontsize=10, backgroundcolor='white')
    ax.text(pred_bbox[0], pred_bbox[1] - 60, f'conf_score: {score}', color='red', fontsize=10, backgroundcolor='white')
    # Draw ground truth bbox in green
    ax.add_patch(patches.Rectangle(
        (gt_bbox[0], gt_bbox[1]), gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
        linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth'
    ))
    plt.legend()
    plt.show()


# --- Main execution ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/Transform_aug.pt"  # Adjust according to checkpoint

model = load_model(model_path, device)
data = load_data()

# takes a random sample and makes a prediction
import random
idx = random.randint(0, len(data)-1)

sample = data[idx]
image = sample["rgb"]
bbox = sample["bbox"]
obj_id = sample["obj_id"]

pred = model_predict(model, image, device)
pred_bbox = pred[0]["boxes"].tolist()
pred_obj_id = pred[0]["labels"].tolist()
pred_score = round(((pred[0]["scores"].tolist())[0]), 3)

visualize(image, pred_bbox[0], pred_obj_id[0], pred_score, bbox)