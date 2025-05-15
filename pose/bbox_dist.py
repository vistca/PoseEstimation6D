
import matplotlib.pyplot as plt
import json
import os
import numpy as np


dirs = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "14", "15"]

info_path = "./datasets/Linemod_preprocessed/data/"
json_name = "gt.json"


heights = []
widths = []
x_centers = []
y_centers = []

save = ""

for dir in dirs:
    pose_data = {}
    json_path = os.path.join(info_path, 'data', dir, "gt.json")

    with open(info_path + dir + "/" + json_name, 'r') as f:
        pose_data = json.load(f)
        for key in pose_data.keys():
            data_point = pose_data[key][0]

            # [x_left, y_top, x_width, y_height]
            bbox = data_point["obj_bb"]

            heights.append(bbox[3])
            widths.append(bbox[2])
            x_centers.append(bbox[0] + 0.5 * bbox[2])
            y_centers.append(bbox[1] + 0.5 * bbox[3])

            if bbox[3] > 250 and save == "":
                save = dir + " + " + key



fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # figsize controls the overall size

# Plot the histograms on each subplot
axes[0, 0].hist(heights, bins=10, edgecolor='black')
axes[0, 0].set_title('heights distribution')

axes[0, 1].hist(widths, bins=10, edgecolor='black')
axes[0, 1].set_title('widths distribution')

axes[1, 0].hist(x_centers, bins=10, edgecolor='black')
axes[1, 0].set_title('x centers distribution')

axes[1, 1].hist(y_centers, bins=10, edgecolor='black')
axes[1, 1].set_title('y centers distribution')

plt.show()

print(f"mean height: {np.mean(heights)}")
print(f"mean width: {np.mean(widths)}")
print(f"special one: {save}")

# {'0': [{'cam_R_m2c': [0.0963063, 0.99404401, 0.0510079, 0.57332098, -0.0135081, -0.81922001, -0.81365103, 0.10814, -0.57120699], 
# 'cam_t_m2c': [-105.3577515, -117.52119142, 1014.8770132], 'obj_bb': [244, 150, 44, 58], 'obj_id': 1}],