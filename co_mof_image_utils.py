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
