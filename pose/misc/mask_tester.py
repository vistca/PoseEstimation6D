from PIL import Image, ImageDraw
import json
import os
import time


dir = "08"
nr = "230"
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

# [x_left, y_top, x_width, y_height]
b = pose_data[nr][0]["obj_bb"] # bounding box

black_bg = Image.new("RGB", img.size, (0, 0, 0))

# Create a mask where the area to keep is white (255), rest is black (0)
mask = Image.new("L", img.size, 0)
draw = ImageDraw.Draw(mask)

print(b)

# Define rectangle to keep (x1, y1, x2, y2)
rectangle = [b[0], b[1], b[2] + b[0], b[3] + b[1]]
draw.rectangle(rectangle, fill=255)

# Composite the original image onto black using the mask
result = Image.composite(img, black_bg, mask)



#draw.rectangle((b[0], b[1], b[0]+b[2], b[1]+b[3]), width=2)

img.show()
result.show()

end = time.perf_counter()



print(end - start)

print(b)



