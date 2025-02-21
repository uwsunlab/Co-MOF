import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.draw import polygon
from skimage.morphology import convex_hull_image

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import pandas as pd


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
        cv2.circle(overlay_image, (int(x1), int(y1)), 5, (255, 0, 0), -1)  # Start point in blue
        cv2.circle(overlay_image, (int(x2), int(y2)), 5, (255, 0, 0), -1)  # End point in yellow

    # Draw all other combined lines in green
    for (x1, y1, x2, y2) in combined_lines:
        if (x1, y1, x2, y2) != longest_line:
            cv2.line(overlay_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green lines

    return overlay_image, longest_line

# Function to display the overlay
def display_overlay(overlay_image):  # TODO: Not sure that this function is needed / is incomplete
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_image)
    plt.title('Binary Image with Combined Horizontal Lines (Longest in Red, Points Highlighted)')
    plt.axis('off')
    plt.show()


# New Code Block

# Function to apply morphological closing
def apply_morphological_closing(bin_image):
    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Apply morphological closing
    closed_image = cv2.morphologyEx(bin_image.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return closed_image

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

# Function to normalize the result and find contours
def normalize_and_find_contours(closing, contour_level=0.8):
    """Normalize the closing result and find contours."""
    
    # Normalize the closing result to values between 0 and 1
    normalized_closing = closing / 255.0

    # Find contours at a constant value (e.g., 0.8)
    contours = measure.find_contours(normalized_closing, contour_level)

    return contours

def filter_and_remove_white_region_contours(closing, contours):
    """Filter and remove contours that enclose white regions."""
    filtered_contours = []
    for contour in contours:
        if not is_contour_enclosing_white_region(contour, closing):
            filtered_contours.append(contour)
    return filtered_contours


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
    boundary_contours = []

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

        # Determine if contour is a boundary (contained within another contour)
        is_boundary = any(
            np.all(contour_masks[j][mask]) for j in range(len(contours)) if j != i
        )

        # Classify based on convexity or add to boundary list
        if is_boundary:
            boundary_contours.append(contour)
        elif convexity > convexity_threshold:
            blue_contours.append(contour)
        else:
            red_contours.append(contour)

    return blue_contours, red_contours, boundary_contours

# Function to display the original color image with classified contours overlaid
def display_classified_contours_on_original_color(original_image, blue_contours, red_contours, boundary_contours):
    """Display the original color image with classified contours overlaid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)  # Display the original color image

    # Plot blue contours
    for contour in blue_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='blue', label="Blue (Convex)")

    # Plot red contours
    for contour in red_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red', label="Red (Non-Convex)")

    # Plot boundary contours in green
    for contour in boundary_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='green', linestyle='--', label="Boundary")

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Classified Contours (Blue: Convex, Red: Non-Convex, Green: Boundary)")
    plt.show()
    
# New Code Block

def overlay_min_bounding_box_and_aspect_ratio(original_image, contours, label_color):
    """
    Overlay the minimum bounding box for contours on the original image and calculate the aspect ratio and area.

    Parameters:
    - original_image: Original color image.
    - contours: List of contours to process.
    - label_color: Color for the label and box.

    Returns:
    - aspect_ratios: List of aspect ratios calculated for the contours.
    - areas: List of areas calculated for the contours.
    """
    aspect_ratios = []
    areas = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    for contour in contours:
        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Find the minimum bounding rectangle
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Draw the bounding rectangle
        ax.plot(
            [box[i][1] for i in range(4)] + [box[0][1]],
            [box[i][0] for i in range(4)] + [box[0][0]],
            color=label_color,
            linewidth=2,
        )

        # Calculate width, height, aspect ratio, and area
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        if height > 0 and width > 0:  # Avoid division by zero
            aspect_ratio = max(width / height, height / width)
            
            if aspect_ratio > 20:
                continue
            aspect_ratios.append(aspect_ratio)

            area = cv2.contourArea(contour_int)
            areas.append(area)

            # Overlay aspect ratio and area
            x_center = np.mean([box[i][1] for i in range(4)])
            y_center = np.mean([box[i][0] for i in range(4)])
            ax.text(
                x_center,
                y_center,
                f"Area: {area:.2f}\nAR: {aspect_ratio:.2f}",
                color=label_color,
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Bounding Box and Aspect Ratio ({label_color.capitalize()})")
    plt.tight_layout()
    plt.show()

    return aspect_ratios, areas


def process_and_overlay_if_few_contours(original_image, blue_contours, red_contours, boundary_contours, length_per_pixel):
    """
    Check if the sum of contours is less than 3. If so, overlay minimum bounding boxes
    and calculate aspect ratios for the corresponding contours.

    Parameters:
    - original_image: Original color image.
    - blue_contours: List of blue (convex) contours.
    - red_contours: List of red (non-convex) contours.
    - boundary_contours: List of boundary contours.

    Returns:
    - all_aspect_ratios: Combined list of all aspect ratios calculated for the contours.
    - all_areas: Combined list of all areas calculated for the contours.
    """
    total_contours = len(blue_contours) + len(red_contours) + len(boundary_contours)
    all_aspect_ratios = []
    all_areas = []

    if total_contours <= 2:
        print(f"Total contours ({total_contours}) are less than or equal to 2. Processing bounding boxes.")
        if blue_contours:
            blue_ar, blue_areas = overlay_min_bounding_box_and_aspect_ratio(
                original_image, blue_contours, "blue"
            )
            blue_areas = blue_areas * length_per_pixel
            all_aspect_ratios.extend(blue_ar)
            all_areas.extend(blue_areas)
        if red_contours:
            red_ar, red_areas = overlay_min_bounding_box_and_aspect_ratio(
                original_image, red_contours, "red"
            )
            red_areas = red_areas * length_per_pixel
            all_aspect_ratios.extend(red_ar)
            all_areas.extend(red_areas)
        if boundary_contours:
            green_ar, green_areas = overlay_min_bounding_box_and_aspect_ratio(
                original_image, boundary_contours, "green"
            )
            all_aspect_ratios.extend(green_ar)
            all_areas.extend(green_areas)
    else:
        print(f"Total contours ({total_contours}) are sufficient. Skipping bounding box overlay.")

    return all_aspect_ratios, all_areas

# New Code Block

def process_blue_contours(image_gray, original_image, blue_contours):
    """Calculate areas and aspect ratios for blue contours and draw bounding rectangles."""
    
    # Lists to store areas and aspect ratios
    areas = []
    aspect_ratios = []

    # Display the original color image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    for idx, contour in enumerate(blue_contours):
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
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Draw the bounding rectangle
        ax.plot(
            [box[i][1] for i in range(4)] + [box[0][1]],
            [box[i][0] for i in range(4)] + [box[0][0]],
            color='blue',
            linewidth=1,
        )

        # Calculate aspect ratio and append to the list
        width, height = rect[1]
        if height > 0 and width > 0:
            aspect_ratio = max(width / height, height / width)
            
            # if aspect_ratio >20:
            #     continue
        else:
            aspect_ratio = 0  # Assign default value for invalid aspect ratio
        aspect_ratios.append(aspect_ratio)
        
        # # Calculate the center of the bounding rectangle to overlay aspect ratio
        # x_center = np.mean([box[i][1] for i in range(4)])
        # y_center = np.mean([box[i][0] for i in range(4)])
        
        # # Overlay aspect ratio on the image
        # ax.text(
        #     x_center,
        #     y_center,
        #     f"{aspect_ratio:.2f}",
        #     color='blue',
        #     fontsize=8,
        #     ha='center',
        #     va='center',
        #     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
        # )

        # Debugging: Print current index, area, and aspect ratio
        # print(f"Contour {idx}: Area = {area}, Aspect Ratio = {aspect_ratio}")

    # Final debug check
    print(f"Total areas: {len(areas)}, Total aspect ratios: {len(aspect_ratios)}")
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Blue Contours with Bounding Rectangles")
    plt.show()

    return areas, aspect_ratios

# New Code Block

# Function to plot top-and-bottom histograms with KDE curves, identify peaks, and overlay standard deviation
def plot_top_bottom_histograms_with_peak_curve_and_std(areas, aspect_ratios):
    """Plot top and bottom histograms with KDE curves, identify peaks, and overlay standard deviation."""
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1]})

    # Top histogram: Areas
    axs[0].hist(areas, bins=200, color='blue', alpha=0.7, density=True, label="Histogram")
    # Calculate KDE curve
    kde_area = gaussian_kde(areas)
    x_area = np.linspace(min(areas), max(areas), 1000)
    y_area = kde_area(x_area)
    axs[0].plot(x_area, y_area, 'r-', label="KDE Curve")
    # Identify the peak
    peak_area = x_area[np.argmax(y_area)]
    axs[0].axvline(peak_area, color='black', linestyle='--', label=f"Peak: {peak_area:.2f}")
    # Overlay standard deviation
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    axs[0].axvline(mean_area - std_area, color='purple', linestyle='--', label=f"Mean - Std Dev: {mean_area - std_area:.2f}")
    axs[0].axvline(mean_area + std_area, color='orange', linestyle='--', label=f"Mean + Std Dev: {mean_area + std_area:.2f}")
    axs[0].set_xlabel('Area')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Histogram of Blue Contour Areas with Peak Curve and Std Dev')
    axs[0].legend()

    # Bottom histogram: Aspect Ratios
    axs[1].hist(aspect_ratios, bins=200, color='green', alpha=0.7, density=True, label="Histogram")
    # Calculate KDE curve
    kde_ar = gaussian_kde(aspect_ratios)
    x_ar = np.linspace(min(aspect_ratios), max(aspect_ratios), 1000)
    y_ar = kde_ar(x_ar)
    axs[1].plot(x_ar, y_ar, 'r-', label="KDE Curve")
    # Identify the peak
    peak_ar = x_ar[np.argmax(y_ar)]
    axs[1].axvline(peak_ar, color='black', linestyle='--', label=f"Peak: {peak_ar:.2f}")
    # Overlay standard deviation
    mean_ar = np.mean(aspect_ratios)
    std_ar = np.std(aspect_ratios)
    axs[1].axvline(mean_ar - std_ar, color='purple', linestyle='--', label=f"Mean - Std Dev: {mean_ar - std_ar:.2f}")
    axs[1].axvline(mean_ar + std_ar, color='orange', linestyle='--', label=f"Mean + Std Dev: {mean_ar + std_ar:.2f}")
    axs[1].set_xlabel('Aspect Ratio')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Histogram of Blue Contour Aspect Ratios with Peak Curve and Std Dev')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return peak_area, peak_ar, mean_area, std_area, mean_ar, std_ar

# New Code Block

def filter_and_overlay_all_contours_with_upper_limit(  # TODO: function is too long
    original_image, image_gray, blue_contours, red_contours, green_contours, mean_area, std_dev_area, peak_area, length_per_pixel
):
    """Filter blue, red, and green (boundary) contours based on area range and overlay them."""
    filtered_blue_contours = []
    filtered_red_contours = []
    filtered_green_contours = []
    blue_aspect_ratios = []  # List to store aspect ratios of blue contours
    blue_contour_areas = []
    upper_limit = mean_area + std_dev_area + 300000  # Value 50000 from trial and error
    min_area = mean_area  # Minimum area as mean_area or peak_area (if there are a lot of impurities)

    img_height, img_width = image_gray.shape  # Image dimensions

    # Overlay the filtered contours
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    # Filter blue contours based on area range
    for contour in blue_contours:
        # Create a binary mask for the contour
        mask = np.zeros(image_gray.shape, dtype=bool)
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image_gray.shape)
        mask[rr, cc] = True

        # Calculate area
        area = np.sum(mask)

        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Find the minimum bounding rectangle
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Check if the bounding box touches the edge of the image
        touches_edge = any(
            (point[1] <= 0 or point[1] >= img_width - 1 or point[0] <= 0 or point[0] >= img_height - 1)
            for point in box
        )

        # Keep blue contours with an area in the valid range and not touching the edges
        if min_area <= area <= upper_limit and not touches_edge:

            # Draw the bounding rectangle
            ax.plot(
                [box[i][1] for i in range(4)] + [box[0][1]],
                [box[i][0] for i in range(4)] + [box[0][0]],
                color="blue",
                linewidth=1,
            )

            # Calculate aspect ratio
            width, height = rect[1]
            aspect_ratio = max(width / height, height / width) if height > 0 and width > 0 else 0
            
            if aspect_ratio > 20:
                continue
            
            # Append valid aspect ratio to the list
            blue_aspect_ratios.append(aspect_ratio)
            
            filtered_blue_contours.append(contour)
            blue_contour_areas.append(area)  # Store the area of the contour
            
            area = area * length_per_pixel

            # # Calculate center of the bounding box
            # x_center = np.mean([box[i][1] for i in range(4)])
            # y_center = np.mean([box[i][0] for i in range(4)])

            # # Overlay area and aspect ratio on the image
            # ax.text(
            #     x_center,
            #     y_center,
            #     f"Area:{area:.0f}\n AR:{aspect_ratio:.2f}",
            #     color="blue",
            #     fontsize=8,
            #     ha="center",
            #     va="center",
            #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            # )
            
            # Calculate a position beside the bounding box (top-left corner of the box)
            top_left_x = np.min([point[1] for point in box])
            top_left_y = np.min([point[0] for point in box])

            # Overlay the area and aspect ratio beside the bounding box
            ax.text(
                top_left_x+100,  # Slightly left of the box
                top_left_y,  # Slightly above the box
                f"Area: {area:.2f}\nAR: {aspect_ratio:.2f}",
                color="blue",
                fontsize=8,
                ha="right",  # Align text to the right
                va="bottom",  # Align text at the bottom
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
            
            

    # Filter red contours based on area range
    for contour in red_contours:
        mask = np.zeros(image_gray.shape, dtype=bool)
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image_gray.shape)
        mask[rr, cc] = True
        area = np.sum(mask)
        if area >= min_area:
            filtered_red_contours.append(contour)

    # Filter green contours (boundary) based on area range
    for contour in green_contours:
        mask = np.zeros(image_gray.shape, dtype=bool)
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=image_gray.shape)
        mask[rr, cc] = True
        filtered_green_contours.append(contour)

    # Plot filtered blue contours
    for contour in filtered_blue_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="blue", label="Blue Contours")

    # Plot filtered red contours
    for contour in filtered_red_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red", label="Red Contours")

    # Plot filtered green contours
    for contour in filtered_green_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="green", label="Green Contours")

    min_area = min_area * length_per_pixel
    upper_limit = upper_limit * length_per_pixel
    
    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Filtered Contours with {min_area:.2f} ≤ Area ≤ {upper_limit:.2f} in µm²")
    plt.show()

    return filtered_blue_contours, blue_contour_areas, filtered_red_contours, filtered_green_contours, blue_aspect_ratios

# New Code Block

def remove_edge_touching_contours_and_display(contours, image_gray, original_image, color):
    """Remove contours that touch the edges of the image and display them."""
    img_height, img_width = image_gray.shape
    filtered_contours = []
    edge_touching_contours = []  # Store contours that are removed

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)
    
    for contour in contours:
        contour_int = np.array(contour, dtype=np.int32)
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Check if the bounding box touches the edge of the image
        touches_edge = any(
            (point[1] <= 0 or point[1] >= img_width - 1 or point[0] <= 0 or point[0] >= img_height - 1)
            for point in box
        )

        if not touches_edge:
            filtered_contours.append(contour)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red", label=f"Filtered {color} Contours")
        else:
            edge_touching_contours.append(contour)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="yellow", label=f"Removed {color} Contours")

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{color.capitalize()} Contours: Filtered vs Removed (Edge Touching)")
    plt.show()

    return filtered_contours

# New Code Block

def plot_bounding_boxes_and_calculate_contour_area(contours, original_image, mean_area, length_per_pixel, color="red"):
    """
    Plot the minimum bounding boxes for filtered contours and calculate their actual contour areas.
    
    Parameters:
    - contours: List of contours to process.
    - original_image: The original image for overlay.
    - length_per_pixel: Conversion factor to get area in real units.
    - color: Color for plotting bounding boxes.
    
    Returns:
    - contour_areas: List of actual contour areas.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    contour_areas = []
    used_positions = []  # List to store positions of previous text labels to avoid overlap
    min_area = mean_area*length_per_pixel

    for contour in contours:
        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Get the minimum area bounding box
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # Convert to integer

        # Draw the bounding box on the plot
        ax.plot([box[i][1] for i in range(4)] + [box[0][1]],
                [box[i][0] for i in range(4)] + [box[0][0]],
                color=color, linewidth=1)

        # Calculate the actual area of the contour
        area = cv2.contourArea(contour_int) * length_per_pixel  # Convert to real-world units
        contour_areas.append(area)
        
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
    ax.set_title(f"Filtered Contours with Area ≥ {min_area:.2f} in µm²")
    plt.show()

    return contour_areas

# New Code Block

def plot_bounding_boxes_and_display_contour_areas(contours, original_image, length_per_pixel, mean_area, color="red"):
    """
    Plot the minimum bounding boxes for filtered contours, calculate actual contour areas, 
    and overlay the area values at a non-overlapping corner of the bounding box while ensuring 
    text is not placed at the edges of the image.

    Parameters:
    - contours: List of contours to process.
    - original_image: The original image for overlay.
    - length_per_pixel: Conversion factor to get area in real units.
    - mean_area: Mean area to set minimum area threshold.
    - color: Color for plotting bounding boxes.

    Returns:
    - contour_areas: List of actual contour areas.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    img_height, img_width, _ = original_image.shape  # Get image dimensions
    contour_areas = []
    used_positions = []  # List to store used text positions to avoid overlap
    min_area = mean_area * length_per_pixel  # Convert to real-world units

    for contour in contours:
        # Convert contour to integer format for OpenCV
        contour_int = np.array(contour, dtype=np.int32)

        # Get the minimum area bounding box
        rect = cv2.minAreaRect(contour_int)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # Convert to integer

        # Draw the bounding box on the plot
        ax.plot([box[i][1] for i in range(4)] + [box[0][1]],
                [box[i][0] for i in range(4)] + [box[0][0]],
                color=color, linewidth=1)

        # Calculate the actual area of the contour
        area = cv2.contourArea(contour_int) * length_per_pixel  # Convert to real-world units
        contour_areas.append(area)

        # Possible label positions (at the four bounding box corners)
        possible_positions = [
            (box[0][1] - 10, box[0][0] - 10),  # Top-left
            (box[1][1] + 10, box[1][0] - 10),  # Top-right
            (box[2][1] + 10, box[2][0] + 10),  # Bottom-right
            (box[3][1] - 10, box[3][0] + 10),  # Bottom-left
        ]

        # Choose a position that does not overlap with previous labels and is not at the edge
        for pos_x, pos_y in possible_positions:
            if (30 < pos_x < img_width - 30 and 30 < pos_y < img_height - 30) and \
               all(abs(pos_x - used_x) > 40 and abs(pos_y - used_y) > 20 for used_x, used_y in used_positions):
                selected_x, selected_y = pos_x, pos_y
                used_positions.append((selected_x, selected_y))
                break
        else:
            # If all positions overlap or are at the edge, adjust position to a safe area
            selected_x = min(max(box[2][1], 40), img_width - 40)  # Keep within bounds
            selected_y = min(max(box[2][0], 40), img_height - 40)
            used_positions.append((selected_x, selected_y))

        # Overlay the area beside the bounding box at the selected corner
        ax.text(
            selected_x, selected_y,
            f"Area: {area:.2f} µm²",
            color=color,
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Filtered Contours with Area ≥ {min_area:.2f} µm²")
    plt.show()

    return contour_areas

# new Code Block

def create_summary_table(
    aspect_ratios_single,
    areas_single, areas_clusters
):
    """
    Create a summary table with the count of contours, mean aspect ratios, and mean areas for single crystals,
    overlapping crystals, and clusters.

    Parameters:
    - single_crystals: List of single crystals.
    - overlapping_crystals: List of overlapping crystals.
    - clusters: List of clusters.
    - aspect_ratios_single: List of aspect ratios for single crystals.
    - aspect_ratios_overlapping: List of aspect ratios for overlapping crystals.
    - aspect_ratios_clusters: List of aspect ratios for clusters.
    - areas_single: List of areas for single crystals.
    - areas_overlapping: List of areas for overlapping crystals.
    - areas_clusters: List of areas for clusters.

    Returns:
    - summary_table: Pandas DataFrame containing the summary.
    """
    # Calculate counts
    count_single_crystals = len(aspect_ratios_single)
    # count_overlapping_crystals = len(aspect_ratios_overlapping)
    count_clusters = len(areas_clusters)

    # Calculate mean aspect ratios
    mean_ar_single = round(np.mean(aspect_ratios_single), 2) if aspect_ratios_single else 0
    # mean_ar_overlapping = round(np.mean(aspect_ratios_overlapping), 2) if aspect_ratios_overlapping else 0
    mean_ar_clusters = "-"

    # Calculate mean areas
    mean_area_single = round(np.mean(areas_single), 2) if areas_single else 0
    # mean_area_overlapping = round(np.mean(areas_overlapping), 2) if areas_overlapping else 0
    mean_area_clusters = round(np.mean(areas_clusters), 2) if areas_clusters else 0

    # Calculate totals
    total_count = count_single_crystals 
    total_aspect_ratios = aspect_ratios_single 
    total_mean_ar = round(np.mean(total_aspect_ratios), 2) if total_aspect_ratios else 0
    # total_areas = areas_single + areas_overlapping 
    # total_mean_area = round(np.mean(total_areas), 2) if total_areas else 0

    # Create the summary table
    summary_table = pd.DataFrame({
        "Category": ["Single Crystals", "Clusters"],
        "Count": [count_single_crystals, count_clusters],
        "Mean Aspect Ratio": [mean_ar_single,  mean_ar_clusters],
        "Mean Area (µm²)": [mean_area_single, mean_area_clusters]
    })

    return summary_table

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
        "Category": ["Single Crystals","Clusters"],
        "Count": [count_single_crystals, count_clusters],
        "Mean Aspect Ratio": [mean_ar_single, mean_ar_clusters],
        "Std Aspect Ratio": [std_ar_single,  std_ar_clusters],
        "Mean Area (µm²)": [mean_area_single,  mean_area_clusters],
        "Std Area (µm²)": [std_area_single, std_area_clusters],
    })

    return summary_table
