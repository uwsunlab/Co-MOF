import os
from matplotlib import pyplot as plt
import pandas as pd

from deprecated import get_length_per_pixel
from co_mof_image_utils import apply_rc_thresholding, grayscale_image, load_rgb_image
from co_mof_ocr import ScaleBarDetector


def main(dataset_csv='dataset/mof_dataset.csv'):
    dataset_df = pd.read_csv(dataset_csv)
    for i in range(len(dataset_df)):
        img_path = dataset_df["Image_Path"].loc[i]
        ocr_result = ScaleBarDetector(img_path)
        _, rc_mask = apply_rc_thresholding(grayscale_image(load_rgb_image(img_path)))
        orig_result = get_length_per_pixel(rc_mask)
        print(f'OCR Result = {ocr_result.units_per_pixel}, {orig_result}')

if __name__ == "__main__":
    main()
    