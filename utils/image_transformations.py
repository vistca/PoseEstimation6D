from PIL import Image, ImageDraw, ImageOps

def rgb_crop_img(rgb_img, b): # b is the bounding box for the image
        m = 0.2 # margin
        # b = [x_left, y_top, x_width, y_height]
        x_min = b[0] - m * b[2]
        x_max = b[0] + (m + 1) * b[2]

        y_min = b[1] - m * b[3]
        y_max = b[1] + (m + 1) * b[3]

        crop = rgb_img.crop((x_min , y_min, x_max, y_max))
        return crop 

def rgb_pad_to_square(rgb_img, fill_color=(0, 0, 0)):
    width, height = rgb_img.size
    max_side = max(width, height)
    
    # Calculate padding for each side
    pad_width = (max_side - width) // 2
    pad_height = (max_side - height) // 2

    # (left, top, right, bottom)
    padding = (pad_width, pad_height,
               max_side - width - pad_width,
               max_side - height - pad_height)
    
    return ImageOps.expand(rgb_img, padding, fill=fill_color)

def rgb_mask_background(rgb_img, b):
    black_bg = Image.new("RGB", rgb_img.size, (0, 0, 0))

    # Create a mask where the area to keep is white (255), rest is black (0)
    mask = Image.new("L", rgb_img.size, 0)
    draw = ImageDraw.Draw(mask)

    # Define rectangle to keep (x1, y1, x2, y2) corners of the image
    rectangle = [b[0], b[1], b[2] + b[0], b[3] + b[1]]
    draw.rectangle(rectangle, fill=255)

    # Composite the original image onto black using the mask
    return Image.composite(rgb_img, black_bg, mask)



