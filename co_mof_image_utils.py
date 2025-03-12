import os
import cv2
import numpy as np
import mahotas as mh


def load_rgb_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Updated image_processing function to ensure the correct data type
def grayscale_image(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.uint8)

def apply_rc_thresholding(im_gray):
    # RC thresholding with Mahotas
    rc_thresh = mh.rc(im_gray)
    bin_rc = im_gray > rc_thresh  # binary image
    return rc_thresh, bin_rc

def resize_and_save_as_jpeg(image_path, dst_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = img.shape[:2]

    if width > height:
        new_width = 1000
        new_height = int((1000 / width) * height)
    else:
        new_height = 1000
        new_width = int((1000 / height) * width)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    base_name = os.path.basename(image_path)
    base_name = os.path.splitext(base_name)[0]
    new_filename = f"{base_name}.jpg"

    cv2.imwrite(os.path.join(dst_dir, new_filename), resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print(f"Image saved as: {os.path.join(dst_dir, new_filename)}")
