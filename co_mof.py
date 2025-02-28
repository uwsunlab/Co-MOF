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

        # # Determine if contour is a boundary (contained within another contour)
        # is_boundary = any(
        #     np.all(contour_masks[j][mask]) for j in range(len(contours)) if j != i
        # )

        # Classify based on convexity or add to boundary list
        # if is_boundary:
        #     boundary_contours.append(contour)
        if convexity > convexity_threshold:
            blue_contours.append(contour)
        else:
            red_contours.append(contour)

    return blue_contours, red_contours

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

    # Plot boundary contours in green
    # for contour in boundary_contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='green', linestyle='--', label="Boundary")

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Classified Contours")
    plt.show()
    
# New Code Block

def overlay_min_bounding_box_and_aspect_ratio(original_image, contours, label_color, length_per_pixel):
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

            area = cv2.contourArea(contour_int) * length_per_pixel
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

#TODO (Arthur): Remove the boundary contours from this part
## Probably can add - count the detected contour at this step if it is less than 2 skip to the last step
def process_and_overlay_if_few_contours(original_image, contours, length_per_pixel):
    """
    Check if the sum of contours is less than 3. If so, overlay minimum bounding boxes
    and calculate aspect ratios for the corresponding contours.

    Parameters:
    - original_image: Original color image.
    - contours: List of all contours

    Returns:
    - all_aspect_ratios: Combined list of all aspect ratios calculated for the contours.
    - all_areas: Combined list of all areas calculated for the contours.
    """
    all_aspect_ratios = []
    all_areas = []
    count = len(contours)

    if count <= 2:
        print(f"Total contours {count}) are less than or equal to 2. Processing bounding boxes.")
        
        aspect_ratio, area = overlay_min_bounding_box_and_aspect_ratio(
            original_image, contours, "blue", length_per_pixel
        )
        # blue_areas = np.array(blue_areas) * length_per_pixel
        all_aspect_ratios.extend(aspect_ratio)
        all_areas.extend(area)

    else:
        print(f"Total contours ({count}) are sufficient. Skipping bounding box overlay.")

    return all_aspect_ratios, all_areas

# New Code Block

def process_blue_contours(image_gray, original_image, contours):
    """Calculate areas and aspect ratios for blue contours and draw bounding rectangles."""
    
    # Lists to store areas and aspect ratios
    areas = []
    aspect_ratios = []

    # Display the original color image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    for idx, contour in enumerate(contours):
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
        

    # # Final debug check
    # print(f"Total areas: {len(areas)}, Total aspect ratios: {len(aspect_ratios)}")
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Blue Contours with Bounding Rectangles")
    plt.show()

    return areas, aspect_ratios

# Function to plot top-and-bottom histograms with KDE curves, identify peaks, and overlay standard deviation
def plot_area_histogram_with_peak_curve(areas):
    """Plot area histograms with KDE curves, identify peaks, and overlay standard deviation."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram: Areas
    ax.hist(areas, bins=200, color='blue', alpha=0.7, density=True, label="Histogram")
    # Calculate KDE curve
    kde_area = gaussian_kde(areas)
    x_area = np.linspace(min(areas), max(areas), 1000)
    y_area = kde_area(x_area)
    ax.plot(x_area, y_area, 'r-', label="KDE Curve")
    # Identify the peak
    peak_area = x_area[np.argmax(y_area)]
    ax.axvline(peak_area, color='black', linestyle='--', label=f"Peak: {peak_area:.2f}")
    # Overlay standard deviation
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    ax.axvline(mean_area - std_area, color='purple', linestyle='--', label=f"Mean - Std Dev: {mean_area - std_area:.2f}")
    ax.axvline(mean_area + std_area, color='orange', linestyle='--', label=f"Mean + Std Dev: {mean_area + std_area:.2f}")
    ax.set_xlabel('Area')
    ax.set_ylabel('Density')
    ax.set_title('Histogram of Blue Contour Areas')
    ax.legend()

    plt.tight_layout()
    plt.show()

    return peak_area, mean_area, std_area

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
        f"Area: {area:.2f}\nAR: {aspect_ratio:.2f}",
        color=color,
        fontsize=8,
        ha="right",  # Align text to the right
        va="bottom",  # Align text at the bottom
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

def filter_and_overlay_all_contours_with_upper_limit(  # TODO: function is too long
    original_image, image_gray, blue_contours, red_contours, mean_area, std_dev_area, length_per_pixel
):
    """Filter blue, and red, contours based on area range and overlay them."""
    filtered_blue_contours = []
    filtered_red_contours = []
    blue_aspect_ratios = []  # List to store aspect ratios of blue contours
    blue_contour_areas = []
    upper_limit = mean_area + std_dev_area + 300000  # Value 30000 from trial and error
    min_area = mean_area  # Minimum area as mean_area or peak_area (if there are a lot of impurities)

    img_height, img_width = image_gray.shape  # Image dimensions

    # Overlay the filtered contours
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)

    # Filter blue contours based on area range
    for contour in blue_contours:
        area = calculate_contour_area_in_pixel(contour, image_gray.shape)

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
            area = area * length_per_pixel
            blue_contour_areas.append(area)
            
            # Calculate a position beside the bounding box (top-left corner of the box)
            top_left_x = np.min([point[1] for point in box])
            top_left_y = np.min([point[0] for point in box])

            overlay_area_AND_aspect_ratio(ax, top_left_x, top_left_y, area, aspect_ratio)

    # Filter red contours based on area range
    for contour in red_contours:
        area = calculate_contour_area_in_pixel(contour, image_gray.shape)
        if area >= min_area:
            filtered_red_contours.append(contour)

    # Plot filtered blue contours
    for contour in filtered_blue_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="blue", label="Blue Contours")

    # Plot filtered red contours
    for contour in filtered_red_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red", label="Red Contours")

    min_area = min_area * length_per_pixel
    upper_limit = upper_limit * length_per_pixel
    
    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Filtered Contours with {min_area:.2f} ≤ Area ≤ {upper_limit:.2f} in µm²")
    plt.show()

    return filtered_blue_contours, blue_contour_areas, filtered_red_contours,  blue_aspect_ratios

# New Code Block
def remove_edge_touching_contours_and_display(contours, image_gray, original_image, color):
    """Remove contours that touch the edges of the image and display them."""
    img_height, img_width = image_gray.shape
    filtered_contours = []

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

    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{color.capitalize()} Contours: Filtered vs Removed (Edge Touching)")
    plt.show()

    return filtered_contours

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