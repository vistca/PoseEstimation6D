from torchvision import transforms



def rgb_crop_img(rgb_img, b, m): # b is the bounding box for the image and m is the wanted margin
        # b = [x_left, y_top, x_width, y_height]
        x_min, y_min, x_max, y_max = map(int, b)

        pad_x = int(m * (x_max - x_min))
        pad_y = int(m * (y_max - y_min))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(rgb_img.width, x_max + pad_x)
        y_max = min(rgb_img.height, y_max + pad_y)
        
        rgb_img = rgb_img.crop((x_min , y_min, x_max, y_max))

        return rgb_img