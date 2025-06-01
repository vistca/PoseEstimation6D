
from PIL import Image, ImageDraw, ImageOps
import json
import os
import time


dir = "14"
nr = "230"
img_nr = "0" * (4 - len(nr))
img_nr = img_nr + nr

img_path = "./datasets/Linemod_preprocessed/data/" + dir + "/rgb/"
info_path = "./datasets/Linemod_preprocessed/data/" + dir + "/"
image_name = img_nr + ".png"
json_name = "gt.json"


def pad_to_square(image, fill_color=(0, 0, 0)):
    width, height = image.size
    max_side = max(width, height)
    
    # Calculate padding for each side
    pad_width = (max_side - width) // 2
    pad_height = (max_side - height) // 2

    # (left, top, right, bottom)
    padding = (pad_width, pad_height,
               max_side - width - pad_width,
               max_side - height - pad_height)
    
    return ImageOps.expand(image, padding, fill=fill_color)

img = Image.open(img_path + image_name).convert("RGB")

pose_data = {}

with open(info_path + json_name, 'r') as f:
    pose_data = json.load(f)

start = time.perf_counter()

# [x_left, y_top, x_width, y_height]
b = pose_data[nr][0]["obj_bb"] # bounding box

draw = ImageDraw.Draw(img)
draw.rectangle((b[0], b[1], b[0]+b[2], b[1]+b[3]), width=2)

img.show()

crop = img.crop((b[0], b[1], b[0]+b[2], b[1]+b[3]))

crop = pad_to_square(crop)

resize_format = (300, 300)
crop = crop.resize(resize_format)

end = time.perf_counter()


crop.show()


print(end - start)

print(b)










