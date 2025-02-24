from matplotlib import pyplot as plt
from skimage import measure
import mahotas as mh
import numpy as np
import cv2


# All functions in this file are deprecated
def deprecated_warning(fxn):
    def wrapper(*args, **kwargs):
        print(f'DEPRECATED WARNING - {fxn.__name__} will be deleted on final release')
        return fxn(*args, **kwargs)
    return wrapper

@deprecated_warning
def apply_otsu_thresholding(im_gray):  # DEPRECATED: not used in favor of RC thresholding
    # Otsu thresholding with Mahotas
    otsu_thresh = mh.otsu(im_gray)
    bin_otsu = im_gray > otsu_thresh  # binary image
    return otsu_thresh, bin_otsu

# Display function to show the binary images with threshold values on histogram
@deprecated_warning
def display_binary_images_with_histogram(im_gray, otsu_thresh, bin_otsu, rc_thresh, bin_rc):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Plot histogram with Otsu and RC thresholds
    ax[0, 0].hist(im_gray.ravel(), bins=255, color='gray')
    ax[0, 0].axvline(otsu_thresh, color='blue', linestyle='--', label=f'Otsu Threshold: {otsu_thresh:.2f}')
    ax[0, 0].axvline(rc_thresh, color='red', linestyle='--', label=f'RC Threshold: {rc_thresh:.2f}')
    ax[0, 0].set_title('Histogram with Thresholds')
    ax[0, 0].legend()

    # Display Otsu threshold binary image
    ax[0, 1].imshow(bin_otsu, cmap='gray')
    ax[0, 1].set_title(f'Otsu Threshold Binary Image (Threshold: {otsu_thresh:.2f})')
    ax[0, 1].axis('off')

    # Display RC threshold binary image
    ax[1, 0].imshow(bin_rc, cmap='gray')
    ax[1, 0].set_title(f'RC Threshold Binary Image (Threshold: {rc_thresh:.2f})')
    ax[1, 0].axis('off')

    # Display original grayscale image for reference
    ax[1, 1].imshow(im_gray, cmap='gray')
    ax[1, 1].set_title('Original Grayscale Image')
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def get_distance_per_pixel_using_longest_contour(rc_mask):
    # Detect all contours in the binary image using skimage
    contours = measure.find_contours(rc_mask, level=0.8)

    # Detect horizontal lines in contours
    horizontal_lines = detect_horizontal_lines(contours)

    # Combine horizontal lines that are close to each other
    combined_lines = combine_close_lines(horizontal_lines, pixel_tolerance=5)

    # Overlay combined horizontal lines on the binary image and highlight the longest in red
    overlay_image, longest_line = overlay_horizontal_lines(rc_mask, combined_lines)
    
    
# Function to combine horizontal lines that are within 5 pixels
def combine_close_lines(horizontal_lines, pixel_tolerance=5):
    combined_lines = []

    # Sort lines by their starting y-coordinate
    horizontal_lines.sort(key=lambda line: line[0][1])

    for line in horizontal_lines:
        x1, y1, x2, y2 = *line[0], *line[1]

        if not combined_lines:
            combined_lines.append((x1, y1, x2, y2))
        else:
            # Check proximity with the last combined line
            cx1, cy1, cx2, cy2 = combined_lines[-1]

            if abs(y1 - cy1) <= pixel_tolerance and (min(x2, cx2) - max(x1, cx1)) >= -pixel_tolerance:
                # Merge lines if they are close
                combined_lines[-1] = (min(x1, cx1), min(y1, cy1), max(x2, cx2), max(y2, cy2))
            else:
                # Otherwise, add as a new line
                combined_lines.append((x1, y1, x2, y2))

    return combined_lines


# Function to detect horizontal straight lines for every contour
def detect_horizontal_lines(contours, slope_tolerance=0.01):
    horizontal_lines = []

    # Iterate through contours
    for idx, contour in enumerate(contours):
        # Contour points
        x_coords, y_coords = contour[:, 1], contour[:, 0]

        # Pairwise line segments
        for i in range(len(x_coords) - 1):
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[i + 1], y_coords[i + 1]

            # Avoid division by zero and compute slope
            if abs(x2 - x1) > 1e-6:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = np.inf

            # Check if the line is horizontal (slope close to 0)
            if abs(slope) <= slope_tolerance:
                horizontal_lines.append(((x1, y1), (x2, y2)))

    return horizontal_lines


# Function to overlay combined horizontal lines and highlight the longest in red
def overlay_horizontal_lines(binary_image, combined_lines):
    # Convert binary image to a 3-channel image for visualization
    overlay_image = np.dstack([binary_image * 255] * 3).astype(np.uint8)

    # Find the longest line
    longest_line = None
    if combined_lines:
        longest_line = max(combined_lines, key=lambda line: np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2))
        (x1, y1, x2, y2) = longest_line

        # Highlight the longest line in red
        cv2.line(overlay_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # Red line

        # Plot starting and ending points of the longest line
        cv2.circle(overlay_image, (int(x1), int(y1)), 15, (255, 0, 0), -1)  # Start point in red
        cv2.circle(overlay_image, (int(x2), int(y2)), 15, (0, 0, 255), -1)  # End point in blue

    # Draw all other combined lines in green
    for (x1, y1, x2, y2) in combined_lines:
        if (x1, y1, x2, y2) != longest_line:
            cv2.line(overlay_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)  # Green lines

    return overlay_image, longest_line


# Function to display the overlay
def display_overlay(overlay_image):  # TODO: Not sure that this function is needed / is incomplete
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_image)
    plt.title('Binary Image with Combined Horizontal Lines (Longest in Red, Points Highlighted)')
    plt.axis('off')
    plt.show()
