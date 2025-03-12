import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.draw import polygon
from skimage.morphology import convex_hull_image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import pandas as pd
import random


class MofImageAnalysis:
    def __init__(self, image_path):
        self.image_path = image_path
        self.preprocess_image(self.load_image())
        
    def load_image(self):
        return load_rgb_image(self.image_path)
    
    def preprocess_image(self):
        self.image = grayscale_image(self.image)

def load_rgb_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Updated image_processing function to ensure the correct data type
def grayscale_image(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.uint8)

# Function to apply Otsu and RC thresholding using Mahotas and return the binary images
# DEPRECATED: not used in favor of RC thresholding
def apply_otsu_thresholding(im_gray):
    # Otsu thresholding with Mahotas
    otsu_thresh = mh.otsu(im_gray)
    bin_otsu = im_gray > otsu_thresh  # binary image
    return otsu_thresh, bin_otsu

def apply_rc_thresholding(im_gray):
    # RC thresholding with Mahotas
    rc_thresh = mh.rc(im_gray)
    bin_rc = im_gray > rc_thresh  # binary image
    return rc_thresh, bin_rc

# Display function to show the binary images with threshold values on histogram
# DEPRECATED
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

# New Code Block

# Function to apply morphological closing
def apply_morphological_closing(bin_image):
    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Apply morphological closing
    closed_image = cv2.morphologyEx(bin_image.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return closed_image

# New Code Block

# Display function to show the RC thresholding results before and after morphological closing
def display_rc_closing_results(im_gray, rc_thresh, bin_rc, closed_image):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Display original grayscale image for reference
    ax[0].imshow(im_gray, cmap='gray')
    ax[0].set_title('Original Grayscale Image')
    ax[0].axis('off')

    # Display RC threshold binary image
    ax[1].imshow(bin_rc, cmap='gray')
    ax[1].set_title(f'RC Threshold Binary Image (Threshold: {rc_thresh})')
    ax[1].axis('off')

    # Display RC threshold binary image after morphological closing
    ax[2].imshow(closed_image, cmap='gray')
    ax[2].set_title('After Morphological Closing')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

# New Code Block

def is_contour_enclosing_white_region(contour, image):  # TODO: rename to avoid using ambiguous colors
    """Check if a contour is enclosing a white region."""
    # Create a mask for the contour
    mask = np.zeros(image.shape, dtype=bool)
    rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image.shape)
    mask[rr, cc] = True

    # Check the number of white pixels inside the contour
    white_pixels_inside = np.sum(mask & (image == 255))  # Pixels inside the contour that are white
    total_pixels_inside = np.sum(mask)  # Total number of pixels inside the contour

    # Check if a large portion of the contour is enclosing white pixels
    # If more than 90% of the area inside the contour is white, return True
    return white_pixels_inside / total_pixels_inside > 0.9 if total_pixels_inside > 0 else False

# New Code Block

# Function to normalize the result and find contours
def normalize_and_find_contours(closing, contour_level=0.8):
    """Normalize the closing result and find contours."""
    
    # Normalize the closing result to values between 0 and 1
    normalized_closing = closing / 255.0

    # Find contours at a constant value (e.g., 0.8)
    contours = measure.find_contours(normalized_closing, contour_level)

    return contours

# New Code Block

def filter_and_remove_white_region_contours(closing, contours):
    """Filter and remove contours that enclose white regions."""
    filtered_contours = []
    for contour in contours:
        if not is_contour_enclosing_white_region(contour, closing):
            filtered_contours.append(contour)
    return filtered_contours


# New Code Block

# Function to display the image and the detected contours
def display_contours(closing, contours):  # TODO: closing is very vague for parameter name
    """Display the image with detected contours overlaid."""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(closing, cmap=plt.cm.gray)  # Display the closing result in grayscale

    # Loop through the contours and plot them
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Contours Detected Using measure.find_contours")
    plt.show()
    
# New Code Block

# Function to display the original color image with detected contours overlaid
def display_contours_on_original_color(original_image, contours):
    """Display the original color image with detected contours overlaid."""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)  # Display the original color image

    # Loop through the contours and plot them on the color image
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')  # Overlay contours in red

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Contours Overlaid on Original Color Image")
    plt.show()    

# New Code Block

# Function to calculate convexity and classify contours into blue, red, and boundary
def classify_contours_by_convexity(image_shape, contours, convexity_threshold=0.865):
    """Classify contours based on convexity threshold and identify boundary contours."""
    blue_contours = []
    red_contours = []
    # boundary_contours = []

    # Create masks for each contour to check for containment
    contour_masks = []
    for contour in contours:
        mask = np.zeros(image_shape, dtype=bool)
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image_shape)
        mask[rr, cc] = True
        contour_masks.append(mask)

    # Classify each contour and check for containment within others
    for i, contour in enumerate(contours):
        mask = contour_masks[i]

        # Calculate convex hull and convexity
        hull = convex_hull_image(mask)
        contour_area = np.sum(mask)
        hull_area = np.sum(hull)
        convexity = contour_area / hull_area

        if convexity > convexity_threshold:
            blue_contours.append(contour)
        else:
            red_contours.append(contour)

    return blue_contours, red_contours

# New Code Block

# Function to display the original color image with classified contours overlaid
def display_classified_contours_on_original_color(original_image, blue_contours, red_contours):
    """Display the original color image with classified contours overlaid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)  # Display the original color image

    # Plot blue contours
    for contour in blue_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='blue', label="Blue (Convex)")

    # Plot red contours
    for contour in red_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red', label="Red (Non-Convex)")

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Classified Contours")
    plt.show()

# New Code Block

def calculate_blue_contour_metrics(image_gray, contours):
    """
    Calculate areas and aspect ratios for blue contours.

    Parameters:
    - image_gray: Grayscale version of the image.
    - contours: List of contours to process.

    Returns:
    - bounding_boxes: List of bounding boxes (4 corner points).
    - areas: List of areas calculated for the contours.
    - aspect_ratios: List of aspect ratios calculated for the contours.
    """
    areas = []
    aspect_ratios = []
    bounding_boxes = []

    for contour in contours:
        # Create a binary mask for the contour
        mask = np.zeros(image_gray.shape, dtype=bool)
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image_gray.shape)
        mask[rr, cc] = True

        # Calculate area and append to the list
        area = np.sum(mask)
        areas.append(area)

        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Find the minimum bounding rectangle
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect).astype(np.int32)
        bounding_boxes.append(box)

        # Calculate aspect ratio
        width, height = rect[1]
        if height > 0 and width > 0:
            aspect_ratio = max(width / height, height / width)
        else:
            aspect_ratio = 0  # Assign default value for invalid aspect ratio
        aspect_ratios.append(aspect_ratio)

    return bounding_boxes, areas, aspect_ratios

# New Code Block

def plot_blue_contours(original_image, bounding_boxes):
    """
    Display the original color image with blue contours' bounding rectangles.

    Parameters:
    - original_image: Original color image.
    - bounding_boxes: List of bounding boxes (4 corner points).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    for box in bounding_boxes:
        # Draw the bounding rectangle
        ax.plot(
            [box[i][1] for i in range(4)] + [box[0][1]],
            [box[i][0] for i in range(4)] + [box[0][0]],
            color='blue',
            linewidth=1,
        )

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Blue Contours with Bounding Rectangles")
    plt.show()

# New Code Block

def calculate_area_statistics_for_blue_contours(areas):
    """
    Compute the KDE curve, peak, mean, and standard deviation for contour areas.

    Parameters:
    - areas: List of contour areas.

    Returns:
    - x_area: X-values for KDE curve.
    - y_area: Y-values for KDE curve.
    - peak_area: The area value corresponding to the highest KDE peak.
    - mean_area: The mean of the contour areas.
    - std_area: The standard deviation of the contour areas.
    """
    kde_area = gaussian_kde(areas)
    x_area = np.linspace(min(areas), max(areas), 1000)
    y_area = kde_area(x_area)

    peak_area = x_area[np.argmax(y_area)]
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    return x_area, y_area, peak_area, mean_area, std_area

# New Code Block

def plot_area_histogram_for_blue_contours(areas, x_area, y_area, peak_area, mean_area, std_area):
    """
    Plot the histogram, KDE curve, peak, and standard deviation overlay.

    Parameters:
    - areas: List of contour areas.
    - x_area: X-values for KDE curve.
    - y_area: Y-values for KDE curve.
    - peak_area: The area value corresponding to the highest KDE peak.
    - mean_area: The mean of the contour areas.
    - std_area: The standard deviation of the contour areas.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram
    ax.hist(areas, bins=200, color='blue', alpha=0.7, density=True, label="Histogram")

    # KDE Curve
    ax.plot(x_area, y_area, 'r-', label="KDE Curve")

    # Peak
    ax.axvline(peak_area, color='black', linestyle='--', label=f"Peak: {peak_area:.2f}")

    # Overlay Standard Deviation
    ax.axvline(mean_area - std_area, color='purple', linestyle='--', label=f"Mean - Std Dev: {mean_area - std_area:.2f}")
    ax.axvline(mean_area + std_area, color='orange', linestyle='--', label=f"Mean + Std Dev: {mean_area + std_area:.2f}")

    ax.set_xlabel('Area')
    ax.set_ylabel('Density')
    ax.set_title('Histogram of Blue Contour Areas')
    ax.legend()

    plt.tight_layout()
    plt.show()

# New Code Block

def calculate_contour_area_in_pixel(contour, image_shape):
    """
    Calculate the area of a contour using a boolean mask.

    Parameters:
    - contour: The contour points as a NumPy array of shape (N, 2).
    - image_shape: The shape of the reference image (height, width).

    Returns:
    - area: The calculated area of the contour.
    """
    mask = np.zeros(image_shape, dtype=bool)
    rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image_shape)
    mask[rr, cc] = True
    return np.sum(mask)

# New Code Block

def overlay_area_AND_aspect_ratio(ax, top_left_x, top_left_y, area, aspect_ratio, color="blue"):
    """
    Overlay the area and aspect ratio beside the bounding box in a plot.

    Parameters:
    - ax: Matplotlib axis object where the text will be drawn.
    - top_left_x: X-coordinate of the top-left corner of the bounding box.
    - top_left_y: Y-coordinate of the top-left corner of the bounding box.
    - area: The area of the contour.
    - aspect_ratio: The aspect ratio of the contour.
    - color: Color of the text (default is "blue").
    """
    ax.text(
        top_left_x + 100,  # Slightly right of the box
        top_left_y,  # Slightly above the box
        f"Area: {area:.2f} µm²\nAR: {aspect_ratio:.2f}",
        color=color,
        fontsize=8,
        ha="right",  # Align text to the right
        va="bottom",  # Align text at the bottom
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    
# New Code Block


def filter_contours_by_area_and_edge(blue_contours, red_contours, image_gray, mean_area, std_dev_area, length_per_pixel):
    """
    Filter blue and red contours based on area range and whether they touch the image edges.

    Parameters:
    - blue_contours: List of blue contours.
    - red_contours: List of red contours.
    - image_gray: Grayscale image (used to get dimensions).
    - mean_area: Mean area used as lower threshold.
    - std_dev_area: Standard deviation of area to determine upper threshold.
    - length_per_pixel: Scaling factor to convert pixel area to real-world units.

    Returns:
    - filtered_blue_contours: List of filtered blue contours.
    - blue_contour_areas: List of area values (scaled).
    - blue_aspect_ratios: List of aspect ratios.
    - filtered_red_contours: List of filtered red contours.
    - min_area: Lower bound for valid contour area (scaled).
    - upper_limit: Upper bound for valid contour area (scaled).
    """
    filtered_blue_contours = []
    filtered_red_contours = []
    blue_aspect_ratios = []
    blue_contour_areas = []

    img_height, img_width = image_gray.shape
    upper_limit = mean_area + std_dev_area + 300000  # Adjusted upper limit based on trial and error
    min_area = mean_area  # Minimum area can be set to peak_area if necessary

    # Process blue contours
    for contour in blue_contours:
        area = calculate_contour_area_in_pixel(contour, image_gray.shape)

        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Find the minimum bounding rectangle
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Check if the bounding box touches the edge of the image
        touches_edge = any(
            (point[1] <= 0 or point[1] >= img_width - 1 or point[0] <= 0 or point[0] >= img_height - 1)
            for point in box
        )

        if min_area <= area <= upper_limit and not touches_edge:
            # Calculate aspect ratio
            width, height = rect[1]
            aspect_ratio = max(width / height, height / width) if height > 0 and width > 0 else 0

            if aspect_ratio > 20:
                continue  # Skip extreme aspect ratios

            filtered_blue_contours.append(contour)
            blue_aspect_ratios.append(aspect_ratio)
            blue_contour_areas.append(area * length_per_pixel)

    # Process red contours
    for contour in red_contours:
        area = calculate_contour_area_in_pixel(contour, image_gray.shape)
        if area >= min_area:
            filtered_red_contours.append(contour)

    return (
        filtered_blue_contours,
        blue_contour_areas,
        blue_aspect_ratios,
        filtered_red_contours,
        min_area * length_per_pixel,
        upper_limit * length_per_pixel
    )

# New Code Block

def plot_filtered_blue_contours(original_image, filtered_blue_contours, blue_contour_areas, blue_aspect_ratios, min_area, upper_limit):
    """
    Plot the filtered contours over the original image with bounding boxes and aspect ratio labels.

    Parameters:
    - original_image: Original color image.
    - filtered_blue_contours: List of filtered blue contours.
    - filtered_red_contours: List of filtered red contours.
    - blue_contour_areas: List of blue contour areas.
    - blue_aspect_ratios: List of aspect ratios for blue contours.
    - min_area: Lower bound for valid contour area (scaled).
    - upper_limit: Upper bound for valid contour area (scaled).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    # Plot filtered blue contours
    for contour, area, aspect_ratio in zip(filtered_blue_contours, blue_contour_areas, blue_aspect_ratios):
        contour_int = np.array(contour, dtype=np.int32)
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Draw bounding rectangle
        ax.plot(
            [box[i][1] for i in range(4)] + [box[0][1]],
            [box[i][0] for i in range(4)] + [box[0][0]],
            color="blue",
            linewidth=1,
        )

        # Calculate label position (top-left corner of the box)
        top_left_x = np.min([point[1] for point in box])
        top_left_y = np.min([point[0] for point in box])

        # Overlay aspect ratio and area
        overlay_area_AND_aspect_ratio(ax, top_left_x, top_left_y, area, aspect_ratio)

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Filtered Contours with {min_area:.2f} ≤ Area ≤ {upper_limit:.2f} in µm²")
    plt.show()
    
# New Code Block

def remove_edge_touching_contours(contours, image_gray):
    """
    Remove contours that touch the edges of the image.

    Parameters:
    - contours: List of contours.
    - image_gray: Grayscale image (used to get image dimensions).

    Returns:
    - filtered_contours: List of contours that do not touch the image edges.
    """
    img_height, img_width = image_gray.shape
    filtered_contours = []

    for contour in contours:
        contour_int = np.array(contour, dtype=np.int32)
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Check if the bounding box touches the edge of the image
        touches_edge = any(
            (point[1] <= 0 or point[1] >= img_width - 1 or point[0] <= 0 or point[0] >= img_height - 1)
            for point in box
        )

        if not touches_edge:
            filtered_contours.append(contour)

    return filtered_contours

# New Code Block

def plot_filtered_red_contours(original_image, contours, color):
    """
    Plot contours over the original image.

    Parameters:
    - original_image: Original color image.
    - contours: List of contours to display.
    - color: Color of the plotted contours.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, label=f"Filtered {color} Contours")

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"FIltered {color.capitalize()} Contours")
    plt.show()

# New Code Block

def calculate_contour_areas(contours, length_per_pixel):
    """
    Calculate the actual contour areas from the given contours.

    Parameters:
    - contours: List of contours to process.
    - length_per_pixel: Conversion factor to get area in real units.

    Returns:
    - contour_areas: List of actual contour areas.
    - bounding_boxes: List of bounding box points for each contour.
    """
    contour_areas = []
    bounding_boxes = []

    for contour in contours:
        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Get the minimum area bounding box
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect).astype(np.int32)  # Convert to integer
        bounding_boxes.append(box)

        # Calculate the actual area of the contour
        area = cv2.contourArea(contour_int) * length_per_pixel  # Convert to real-world units
        contour_areas.append(area)

    return contour_areas, bounding_boxes

# New Code Block

def plot_bounding_boxes_with_areas(original_image, bounding_boxes, contour_areas, min_area, length_per_pixel, color="red"):
    """
    Plot the minimum bounding boxes for contours and overlay their actual areas.

    Parameters:
    - original_image: The original image for overlay.
    - bounding_boxes: List of bounding box points.
    - contour_areas: List of actual contour areas.
    - mean_area: Mean contour area for reference.
    - length_per_pixel: Conversion factor to real units.
    - color: Color for plotting bounding boxes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    used_positions = []  # List to store positions of previous text labels to avoid overlap
    # min_area = mean_area * length_per_pixel

    for box, area in zip(bounding_boxes, contour_areas):
        # Draw the bounding box on the plot
        ax.plot([box[i][1] for i in range(4)] + [box[0][1]],
                [box[i][0] for i in range(4)] + [box[0][0]],
                color=color, linewidth=1)

        # Determine a non-overlapping position for displaying the area
        possible_positions = [
            (np.min(box[:, 1]) - 10, np.min(box[:, 0]) - 10),  # Top-left
            (np.max(box[:, 1]) + 10, np.min(box[:, 0]) - 10),  # Top-right
            (np.min(box[:, 1]) - 10, np.max(box[:, 0]) + 10),  # Bottom-left
            (np.max(box[:, 1]) + 10, np.max(box[:, 0]) + 10),  # Bottom-right
        ]

        # Choose a position that does not overlap with existing labels
        for pos_x, pos_y in possible_positions:
            if all(abs(pos_x - used_x) > 40 and abs(pos_y - used_y) > 15 for used_x, used_y in used_positions):
                selected_x, selected_y = pos_x, pos_y
                used_positions.append((selected_x, selected_y))
                break
        else:
            # If all positions overlap, place it near the bottom-right
            selected_x, selected_y = possible_positions[-1]
            used_positions.append((selected_x, selected_y))

        # Overlay the area beside the bounding box
        ax.text(
            selected_x + 50, selected_y - 10,
            f"Area: {area:.2f} µm²",
            color=color,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Filtered Contours with Area ≤ {min_area:.2f} in µm²")
    plt.show()

# New Code Block

# Function to calculate the standard deviation
def calculate_summary_with_std(aspect_ratios_single, areas_single, areas_clusters):
    """
    Calculate summary statistics including standard deviations.

    Parameters are similar to the create_summary_table function.

    Returns:
    - summary_table: Pandas DataFrame containing the summary statistics.
    """
    # Calculate counts
    count_single_crystals = len(aspect_ratios_single)
    count_clusters = len(areas_clusters)

    # Calculate mean and standard deviations
    mean_ar_single = round(np.mean(aspect_ratios_single), 2) if aspect_ratios_single else 0
    std_ar_single = round(np.std(aspect_ratios_single), 2) if aspect_ratios_single else 0

    mean_ar_clusters = "-"
    std_ar_clusters = "-"

    mean_area_single = round(np.mean(areas_single), 2) if areas_single else 0
    std_area_single = round(np.std(areas_single), 2) if areas_single else 0

    mean_area_clusters = round(np.mean(areas_clusters), 2) if areas_clusters else 0
    std_area_clusters = round(np.std(areas_clusters), 2) if areas_clusters else 0

    total_areas = areas_single 
    total_mean_area = round(np.mean(total_areas), 2) if total_areas else 0
    total_std_area = round(np.std(total_areas), 2) if total_areas else 0

    # Create summary table
    summary_table = pd.DataFrame({
        "Category": ["Isolated Crystals","Clusters"],
        "Count": [count_single_crystals, count_clusters],
        "Mean Aspect Ratio": [mean_ar_single, mean_ar_clusters],
        "Std Aspect Ratio": [std_ar_single,  std_ar_clusters],
        "Mean Area (µm²)": [mean_area_single,  mean_area_clusters],
        "Std Area (µm²)": [std_area_single, std_area_clusters],
    })

    return summary_table

# New Code Block

def plot_summary_bar_chart(summary_table, area_scaling_factor=1e3):
    """
    Plots a grouped bar chart for Count, Mean Aspect Ratio, and Mean Area from a summary table.

    Parameters:
    - summary_table: Pandas DataFrame containing "Category", "Count", "Mean Aspect Ratio", and "Mean Area (µm²)".
    - area_scaling_factor: Factor to scale the Mean Area values for better visualization (default: 1e3).
    """
    # Extract data
    categories = summary_table["Category"][::-1]
    counts = summary_table["Count"][::-1]
    mean_aspect_ratios = summary_table["Mean Aspect Ratio"][::-1]
    mean_areas = summary_table["Mean Area (µm²)"][::-1]

    # Define bar positions and width
    bar_width = 0.25
    indices = np.arange(len(categories))

    # Convert Mean Aspect Ratio and Mean Area to floats, handling non-numeric values like "-"
    mean_aspect_ratios = [float(val) if val != "-" else 0 for val in mean_aspect_ratios]
    mean_areas = [float(val) if val != "-" else 0 for val in mean_areas]

    # Normalize Mean Area for better visualization
    mean_areas_scaled = [area / area_scaling_factor for area in mean_areas]

    # Plot grouped vertical bar chart
    fig, ax = plt.subplots(figsize=(8, 8))

    # Add bars for Count
    bars_count = ax.bar(indices - bar_width, counts, width=bar_width, color='thistle', label='Count')

    # Add bars for Mean Aspect Ratio
    bars_aspect = ax.bar(indices, mean_aspect_ratios, width=bar_width, color='steelblue', label='Mean Aspect Ratio')

    # Add bars for Mean Area (scaled)
    bars_area = ax.bar(indices + bar_width, mean_areas_scaled, width=bar_width, color='coral', label=r"Mean Area ($\times10^3$ µm$^2$)")

    # Add text labels
    for bars, data in zip([bars_count, bars_aspect, bars_area], [counts, mean_aspect_ratios, mean_areas_scaled]):
        for bar, value in zip(bars, data):
            if value != 0:  # Ignore labels for bars with a value of 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Center text horizontally on the bar
                    bar.get_height() + 0.05,  # Position slightly above the bar
                    f"{value:.2f}" if isinstance(value, float) else f"{int(value)}",  # Format the text
                    ha='center', fontsize=15
                )

    # Add labels and legend
    ax.set_ylabel("Values", fontsize=17)
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, fontsize=17, rotation=45)
    ax.tick_params(axis='y', labelsize=17)
    ax.legend(fontsize=17, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
    
# -----------------------------------------------
# # Additional code for overlapping crystals detection
# -----------------------------------------------
    
def analyze_isolated_aspect_ratios(blue_aspect_ratios):
    """
    Compute the mean and standard deviation of the aspect ratios of the filtered blue contours.

    Parameters:
    - blue_aspect_ratios: List of aspect ratios for the filtered blue contours.

    Returns:
    - mean_ar: Mean of the blue aspect ratios.
    - std_dev_ar: Standard deviation of the blue aspect ratios.
    """
    # Convert to NumPy array
    blue_aspect_ratios = np.array(blue_aspect_ratios)

    # Compute statistics
    mean_ar = np.mean(blue_aspect_ratios)
    std_dev_ar = np.std(blue_aspect_ratios)

    return mean_ar, std_dev_ar

# New Code Block

def plot_isolated_aspect_ratios(blue_aspect_ratios, mean_ar, std_dev_ar):
    """
    Create a box plot for the aspect ratios of the filtered blue contours.

    Parameters:
    - blue_aspect_ratios: List of aspect ratios for the filtered blue contours.
    - mean_ar: Mean of the blue aspect ratios.
    - std_dev_ar: Standard deviation of the blue aspect ratios.
    """
    plt.figure(figsize=(8, 6))

    # Create the box plot
    boxplot = plt.boxplot(
        blue_aspect_ratios,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor='cyan', color='cyan')
    )

    plt.title("Box Plot of Aspect Ratios for Filtered Blue Contours")
    plt.ylabel("Aspect Ratio")
    plt.xticks([1], ["Blue Contours"])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend with mean and standard deviation
    plt.legend(
        [boxplot["boxes"][0]],
        [f"Mean: {mean_ar:.2f}, SD: {std_dev_ar:.2f}"],
        loc="upper right",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="black"
    )

    plt.tight_layout()
    plt.show()

# New Code Block

def detect_and_group_lines_per_contour(image_grey, red_contours, group_threshold=15):
    """
    Detect and group lines within each red contour based on angle similarity.
    Remove lines shorter than 5 pixels.

    Parameters:
    - binary_background: Binary image used for reference.
    - red_contours: List of red contours to detect lines.
    - group_threshold: The angle threshold in degrees for grouping lines within a contour.

    Returns:
    - grouped_lines_per_contour: A list of lists, where each inner list contains grouped lines for a specific contour.
    """
    def calculate_angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))

    grouped_lines_per_contour = []

    for red_contour in red_contours:
        # Create a mask for the current red contour
        red_mask = np.zeros_like(image_grey, dtype=np.uint8)
        rr, cc = polygon(red_contour[:, 0], red_contour[:, 1], red_mask.shape)
        red_mask[rr, cc] = 255

        # Use Canny edge detection to find edges in the red contour mask
        edges = cv2.Canny(red_mask, 50, 150)

        # Use Probabilistic Hough Line Transform to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=5)

        if lines is not None:
            # Store lines and their angles, filtering out lines shorter than 20 pixels
            line_groups = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Calculate the Euclidean distance
                if length >= 5:  # Filter out lines shorter than 5 pixels
                    angle = calculate_angle(x1, y1, x2, y2)
                    line_groups.append({"line": [x1, y1, x2, y2], "angle": angle})

            # Group lines based on angle similarity within the current contour
            grouped_lines = []
            while line_groups:
                current_line = line_groups.pop(0)
                current_angle = current_line["angle"]
                current_group = [current_line["line"]]
                new_group = []

                # Compare the current line with all other lines in the same contour
                for other_line in line_groups:
                    other_angle = other_line["angle"]

                    # If the angle difference is within the threshold, group them together
                    if abs(current_angle - other_angle) <= group_threshold:
                        current_group.append(other_line["line"])
                    else:
                        new_group.append(other_line)

                # Add the group to the grouped lines
                grouped_lines.append(current_group)
                line_groups = new_group

            # Store the grouped lines for the current contour
            grouped_lines_per_contour.append(grouped_lines)

    return grouped_lines_per_contour

# New Code Block

def plot_grouped_lines_per_contour_on_original_image(original_image, grouped_lines_per_contour):
    """
    Plot the grouped lines for each red contour with different colors for each group on the original image.

    Parameters:
    - original_image: Original image for reference.
    - grouped_lines_per_contour: List of lists, where each inner list contains grouped lines for a specific contour.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Show the original image
    ax.imshow(original_image)

    # Plot each contour's grouped lines with different colors for each group
    for grouped_lines in grouped_lines_per_contour:
        for group in grouped_lines:
            # Generate a random color for each group
            color = (random.random(), random.random(), random.random())  # RGB values between 0 and 1 for matplotlib

            # Plot all lines in the group with the same color
            for line in group:
                x1, y1, x2, y2 = line
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, linestyle='-')

    # Set title and axis settings
    ax.set_title('Grouped Lines by Angle Within Each Contour (Overlaid on Original Image)')
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    
# New Code Block

def analyze_red_contour_aspect_ratios(grouped_lines_per_contour, filtered_red_contours, max_groups=5):
    """
    Compute aspect ratios and bounding boxes for grouped lines in red contours.

    Parameters:
    - grouped_lines_per_contour: List of grouped lines for each contour.
    - filtered_red_contours: List of red contours corresponding to the grouped lines.
    - max_groups: Maximum number of grouped lines per contour.

    Returns:
    - bounding_boxes_data: List of tuples (contour, bounding box, aspect ratio).
    """
    bounding_boxes_data = []

    for contour, grouped_lines in zip(filtered_red_contours, grouped_lines_per_contour):
        if len(grouped_lines) > max_groups:
            continue

        for group in grouped_lines:
            if len(group) < 3:
                continue  # Skip groups with 3 or fewer lines

            # Flatten lines into a point cloud
            points = np.array([[x, y] for line in group for x, y in [(line[0], line[1]), (line[2], line[3])]])

            if points.shape[0] > 0:
                pca = PCA(n_components=2)
                pca.fit(points)
                transformed_points = pca.transform(points)
                min_x, min_y = np.min(transformed_points, axis=0)
                max_x, max_y = np.max(transformed_points, axis=0)

                box_transformed = np.array([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ])
                box = pca.inverse_transform(box_transformed)

                # Calculate aspect ratio
                width = np.linalg.norm(box[0] - box[1])
                height = np.linalg.norm(box[1] - box[2])
                if height > 0:
                    aspect_ratio = max(width / height, height / width)

                    bounding_boxes_data.append((contour, box, aspect_ratio))

    return bounding_boxes_data

# New Code Block

def filter_valid_red_bounding_boxes(image_gray, bounding_boxes_data, min_AR=1.5, max_AR=5.0):
    """
    Filter bounding boxes based on aspect ratio and whether they touch the image edges.

    Parameters:
    - bounding_boxes_data: List of (contour, bounding box, aspect ratio).
    - min_AR: Minimum aspect ratio.
    - max_AR: Maximum aspect ratio.
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - valid_boxes: List of valid bounding boxes.
    - red_aspect_ratios: List of aspect ratios for valid bounding boxes.
    - processed_contours: List of contours corresponding to valid bounding boxes.
    """
    valid_boxes = []
    red_aspect_ratios = []
    processed_contours = []
    
    img_height, img_width = image_gray.shape

    for contour, box, aspect_ratio in bounding_boxes_data:
        if not (min_AR <= aspect_ratio <= max_AR):
            continue

        # Check if the bounding box touches the image edges
        if img_width and img_height:
            touches_edge = any(
                (point[0] <= 0 or point[0] >= img_width - 1 or point[1] <= 0 or point[1] >= img_height - 1)
                for point in box
            )
            if touches_edge:
                continue

        valid_boxes.append(box)
        red_aspect_ratios.append(aspect_ratio)
        processed_contours.append(contour)

    return valid_boxes, red_aspect_ratios, processed_contours

# New Code Block

def generate_yellow_bounding_boxes(image_gray, filtered_red_contours, processed_contours):
    """
    Generate yellow bounding boxes for contours that do not have valid red bounding boxes.

    Parameters:
    - filtered_red_contours: List of all red contours.
    - processed_contours: List of contours that already have valid red bounding boxes.
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - yellow_boxes: List of bounding boxes for remaining contours.
    - unprocessed_contours: List of contours corresponding to yellow boxes.
    """
    yellow_boxes = []
    unprocessed_contours = []
    
    img_height, img_width = image_gray.shape

    for contour in filtered_red_contours:
        if any(np.array_equal(contour, proc_contour) for proc_contour in processed_contours):
            continue  # Skip contours that were already processed

        # Compute the minimum bounding box
        x_coords, y_coords = contour[:, 1], contour[:, 0]
        min_x, min_y = np.min(x_coords), np.min(y_coords)
        max_x, max_y = np.max(x_coords), np.max(y_coords)
        box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

        # Check if the bounding box touches the image edges
        touches_edge = any(
            (point[0] <= 0 or point[0] >= img_width - 1 or point[1] <= 0 or point[1] >= img_height - 1)
            for point in box
        )
        if touches_edge:
            continue

        yellow_boxes.append(box)
        unprocessed_contours.append(contour)

    return yellow_boxes, unprocessed_contours

# New Code Block

def plot_red_and_yellow_bounding_boxes(original_image, valid_boxes, red_aspect_ratios, processed_contours, yellow_boxes):
    """
    Plot the red and yellow bounding boxes.

    Parameters:
    - original_image: The original image for reference.
    - valid_boxes: List of valid red bounding boxes.
    - red_aspect_ratios: List of aspect ratios for red bounding boxes.
    - processed_contours: List of contours corresponding to red boxes.
    - yellow_boxes: List of yellow bounding boxes.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_image)

    # Plot red bounding boxes
    for box in valid_boxes:
        ax.plot(
            [box[i][0] for i in range(4)] + [box[0][0]],
            [box[i][1] for i in range(4)] + [box[0][1]],
            'r-', linewidth=2
        )

    # Plot yellow bounding boxes
    for box in yellow_boxes:
        ax.plot(
            [box[i][0] for i in range(4)] + [box[0][0]],
            [box[i][1] for i in range(4)] + [box[0][1]],
            'y-', linewidth=2
        )

    ax.set_title("Overlapping Crystals (Red) & Clusters (Yellow)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    
# New Code Block

def calculate_overlapping_area(binary_image, box_information, length_per_pixel):
    """
    Calculate the number of black pixels inside each bounding box.

    Parameters:
    - binary_image: Binary image for black pixel counting.
    - box_information: List of bounding box coordinates.
    - length_per_pixel: Conversion factor for real-world units.

    Returns:
    - black_pixel_counts: List of black pixel counts for each bounding box.
    """
    img_height, img_width = binary_image.shape
    black_pixel_counts = []

    for box in box_information:
        # Create a mask for the bounding box
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        box = np.array(box, dtype=np.int32)
        cv2.fillPoly(mask, [box], 255)

        # Extract the region within the bounding box using the mask
        masked_region = cv2.bitwise_and(binary_image, binary_image, mask=mask)

        # Count black pixels in the region
        black_pixel_count = np.sum(masked_region[mask == 255] == 0) * length_per_pixel
        black_pixel_counts.append(black_pixel_count)

    return black_pixel_counts

# New Code Block

def overlay_text_on_boxes(ax, box_information, black_pixel_counts, red_aspect_ratios):
    """
    Overlay text with black pixel count and aspect ratio beside each bounding box.

    Parameters:
    - ax: Matplotlib axis to plot text.
    - box_information: List of bounding box coordinates.
    - black_pixel_counts: List of black pixel counts corresponding to bounding boxes.
    - red_aspect_ratios: List of aspect ratios for bounding boxes.
    """
    for idx, box in enumerate(box_information):
        if idx < len(red_aspect_ratios):
            aspect_ratio = red_aspect_ratios[idx]
            black_pixel_count = black_pixel_counts[idx]

            # Calculate a position beside the bounding box (top-left corner)
            top_left_x = np.min([point[0] for point in box])
            top_left_y = np.min([point[1] for point in box])

            # Overlay the count and aspect ratio beside the bounding box
            ax.text(
                top_left_x,  # Slightly left of the box
                top_left_y,  # Slightly above the box
                f"Area: {black_pixel_count:.2f} µm²\nAR: {aspect_ratio:.2f}",
                color="red",
                fontsize=8,
                ha="right",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

# New Code Block

def plot_overlapping_area_and_aspect_ratios(original_image, box_information, black_pixel_counts, red_aspect_ratios):
    """
    Plot bounding boxes on the original image and overlay black pixel counts & aspect ratios.

    Parameters:
    - original_image: Original image for visualization.
    - box_information: List of bounding box coordinates.
    - black_pixel_counts: List of black pixel counts for bounding boxes.
    - red_aspect_ratios: List of aspect ratios for bounding boxes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the original image
    ax.imshow(original_image)

    # Draw bounding boxes
    for box in box_information:
        ax.plot(
            [box[i][0] for i in range(4)] + [box[0][0]],
            [box[i][1] for i in range(4)] + [box[0][1]],
            "r-",
            linewidth=2,
        )

    # Overlay text annotations
    overlay_text_on_boxes(ax, box_information, black_pixel_counts, red_aspect_ratios)

    # Finalize the plot
    ax.axis("off")
    plt.title("Overlapping Crystals (µm²)")
    plt.tight_layout()
    plt.show()

# New Code Block

def calculate_unprocessed_contour_areas(unprocessed_contours, length_per_pixel):
    """
    Calculate the area of each unprocessed contour.

    Parameters:
    - unprocessed_contours: List of unprocessed contours.
    - length_per_pixel: Conversion factor for real-world units.

    Returns:
    - areas: List of contour areas in real-world units.
    - bounding_boxes: List of bounding box coordinates for visualization.
    """
    areas = []
    bounding_boxes = []

    for contour in unprocessed_contours:
        # Calculate the minimum bounding box
        x_coords = contour[:, 1]
        y_coords = contour[:, 0]
        min_x, min_y = np.min(x_coords), np.min(y_coords)
        max_x, max_y = np.max(x_coords), np.max(y_coords)
        box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])
        
        # Calculate contour area
        contour_area = cv2.contourArea(contour.astype(np.int32)) * length_per_pixel
        areas.append(contour_area)
        bounding_boxes.append(box)

    return areas, bounding_boxes

# New Code Block

def plot_unprocessed_contour_areas(original_image, bounding_boxes, areas):
    """
    Overlay bounding boxes and areas of unprocessed contours onto the original image.

    Parameters:
    - original_image: The original image for reference.
    - bounding_boxes: List of bounding box coordinates.
    - areas: List of contour areas.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert image if necessary
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)

    # Show the original image
    ax.imshow(original_image)

    for idx, box in enumerate(bounding_boxes):
        area = areas[idx]

        # Calculate position for text overlay
        top_left_x = np.min([point[0] for point in box])
        top_left_y = np.min([point[1] for point in box])

        # Overlay area text
        ax.text(
            top_left_x, top_left_y, f"Area: {area:.2f} µm²",
            color="yellow", fontsize=8, ha="right", va="bottom",
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none")
        )

        # Plot the bounding box
        ax.plot(
            [box[i][0] for i in range(4)] + [box[0][0]],
            [box[i][1] for i in range(4)] + [box[0][1]],
            'y-', linewidth=2, label='Yellow Bounding Box'
        )

    ax.set_title('Clusters (µm²)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# New Code Block

def OC_calculate_summary_with_std(aspect_ratios_single, aspect_ratios_overlapping, areas_single, areas_overlapping, areas_clusters):
    """
    Calculate summary statistics including standard deviations.

    Parameters are similar to the create_summary_table function.

    Returns:
    - summary_table: Pandas DataFrame containing the summary statistics.
    """
    # Calculate counts
    count_single_crystals = len(aspect_ratios_single)
    count_overlapping_crystals = len(aspect_ratios_overlapping)
    count_clusters = len(areas_clusters)

    # Calculate mean and standard deviations
    mean_ar_single = round(np.mean(aspect_ratios_single), 2) if aspect_ratios_single else 0
    std_ar_single = round(np.std(aspect_ratios_single), 2) if aspect_ratios_single else 0

    mean_ar_overlapping = round(np.mean(aspect_ratios_overlapping), 2) if aspect_ratios_overlapping else 0
    std_ar_overlapping = round(np.std(aspect_ratios_overlapping), 2) if aspect_ratios_overlapping else 0

    mean_ar_clusters = "-"
    std_ar_clusters = "-"

    mean_area_single = round(np.mean(areas_single), 2) if areas_single else 0
    std_area_single = round(np.std(areas_single), 2) if areas_single else 0

    mean_area_overlapping = round(np.mean(areas_overlapping), 2) if areas_overlapping else 0
    std_area_overlapping = round(np.std(areas_overlapping), 2) if areas_overlapping else 0

    mean_area_clusters = round(np.mean(areas_clusters), 2) if areas_clusters else 0
    std_area_clusters = round(np.std(areas_clusters), 2) if areas_clusters else 0

    total_count = count_single_crystals + count_overlapping_crystals
    total_aspect_ratios = aspect_ratios_single + aspect_ratios_overlapping
    total_mean_ar = round(np.mean(total_aspect_ratios), 2) if total_aspect_ratios else 0
    total_std_ar = round(np.std(total_aspect_ratios), 2) if total_aspect_ratios else 0

    total_areas = areas_single + areas_overlapping
    total_mean_area = round(np.mean(total_areas), 2) if total_areas else 0
    total_std_area = round(np.std(total_areas), 2) if total_areas else 0

    # Create summary table
    summary_table = pd.DataFrame({
        "Category": ["Isolated Crystals", "Overlapping Crystals", "Clusters"],
        "Count": [count_single_crystals, count_overlapping_crystals, count_clusters],
        "Mean Aspect Ratio": [mean_ar_single, mean_ar_overlapping, mean_ar_clusters],
        "Std Aspect Ratio": [std_ar_single, std_ar_overlapping,  std_ar_clusters],
        "Mean Area (µm²)": [mean_area_single, mean_area_overlapping,  mean_area_clusters],
        "Std Area (µm²)": [std_area_single, std_area_overlapping, std_area_clusters],
    })

    return summary_table

# New Code Block

def OC_plot_summary_bar_chart(summary_table_with_std, area_scaling_factor=1e3):
    """
    Generate a grouped bar chart for Counts, Mean Aspect Ratios, and Mean Areas 
    for different categories in the summary table.

    Parameters:
    - summary_table_with_std: Pandas DataFrame containing summary statistics.
    - area_scaling_factor: Factor to scale the Mean Area for better visualization (default=1e3).

    Returns:
    - None (displays the plot)
    """
    
    # Extract data from the summary table
    categories = summary_table_with_std["Category"][::-1]
    counts = summary_table_with_std["Count"][::-1]
    mean_aspect_ratios = summary_table_with_std["Mean Aspect Ratio"][::-1]
    mean_areas = summary_table_with_std["Mean Area (µm²)"][::-1]

    # Define bar positions and width
    bar_width = 0.25
    indices = np.arange(len(categories))

    # Convert Mean Aspect Ratio and Mean Area to floats, handling non-numeric values like "-"
    mean_aspect_ratios = [float(val) if val != "-" else 0 for val in mean_aspect_ratios]
    mean_areas = [float(val) if val != "-" else 0 for val in mean_areas]

    # Normalize Mean Area for better visualization alongside Count and Aspect Ratio
    mean_areas_scaled = [area / area_scaling_factor for area in mean_areas]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add bars for Count
    bars_count = plt.bar(indices - bar_width, counts, width=bar_width, color='thistle', label='Count')

    # Add bars for Mean Aspect Ratio
    bars_aspect = plt.bar(indices, mean_aspect_ratios, width=bar_width, color='steelblue', label='Mean Aspect Ratio')

    # Add bars for Mean Area (scaled)
    bars_area = plt.bar(indices + bar_width, mean_areas_scaled, width=bar_width, color='coral', label=r"Mean Area ($\times10^3$ µm$^2$)")

    # Add text labels above bars
    for bars, data in zip([bars_count, bars_aspect, bars_area], [counts, mean_aspect_ratios, mean_areas_scaled]):
        for bar, value in zip(bars, data):
            if value != 0:  # Ignore labels for bars with a value of 0
                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # Center the text horizontally on the bar
                    bar.get_height() + 0.05,  # Position slightly above the bar
                    f"{value:.2f}" if isinstance(value, float) else f"{int(value)}",  # Format the text
                    ha='center', fontsize=15
                )

    # Add labels and legend
    plt.ylabel("Values", fontsize=17)
    plt.xticks(indices, categories, fontsize=17, rotation=45)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17, loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

