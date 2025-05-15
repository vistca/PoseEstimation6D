
from PIL import Image, ImageDraw
import json
import os
import time


dir = "09"
nr = "100"
img_nr = "0" * (4 - len(nr))
img_nr = img_nr + nr

img_path = "./datasets/Linemod_preprocessed/data/" + dir + "/rgb/"
info_path = "./datasets/Linemod_preprocessed/data/" + dir + "/"
image_name = img_nr + ".png"
json_name = "gt.json"


img = Image.open(img_path + image_name).convert("RGB")

pose_data = {}

with open(info_path + json_name, 'r') as f:
    pose_data = json.load(f)


start = time.perf_counter()

b = pose_data[nr][0]["obj_bb"] # bounding box


crop = img.crop((b[0], b[1], b[0]+b[2], b[1]+b[3]))

resize_format = (64, 64)
crop = crop.resize(resize_format)

end = time.perf_counter()


crop.show()


print(end - start)





