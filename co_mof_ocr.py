import easyocr
import re
import cv2
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt

from co_mof_image_utils import load_rgb_image, grayscale_image, apply_rc_thresholding


class ScaleBarDetector:
    def __init__(self, image_src: str, u2μ=True):
        self.image_src = image_src
        self.roi_bbox = get_bottom_right_bbox(load_rgb_image(self.image_src))
        self.physical_length = None
        self.units = None
        self.pixel_length = None
        self.confidence = None
        self.text_bbox = None
        self.scale_bar_bbox = None
        self.units_per_pixel = None
        self.set_scale_bar_attributes(u2μ=u2μ)

    def set_scale_bar_attributes(self, u2μ=True):
        """
        Reads the image using EasyOCR, searches for a string that consists of a number followed by letters 
        (representing any unit), and returns the best match (based on OCR confidence) with the bounding box,
        the numeric value, and the unit.
        
        Parameters:
            image_path (str): Path to the image file.
            
        Returns:
            dict or None: A dictionary with keys 'bounding_box', 'number', 'unit', and 'confidence' for the best match,
                        or None if no match is found.
        """
        roi_image = crop_from_bbox(load_rgb_image(self.image_src), self.roi_bbox)
        
        reader = easyocr.Reader(['en'])
        results = reader.readtext(roi_image)
        
        # Regex pattern: one or more digits (the numerical length), optional whitespace, followed by one or more letters (the units)
        pattern = r'\b(\d+)\s*([a-zA-Z]+)\b'
        
        best_match = None
        highest_confidence = 0
        
        for bbox, text, confidence in results:
            found = re.findall(pattern, text)
            if found:
                number_str, units = found[0]
                if u2μ and units.startswith('u'):
                    units = 'μ' + units[1:]
                match = {
                    'bounding_box': bbox,   # List of four (x, y) coordinates.
                    'physical_length': int(number_str),  # Convert the numeric string to an integer.
                    'units': units,  # The extracted unit (e.g., "um", "nm", etc.)
                    'confidence': confidence
                }
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = match
        
        if best_match:
            self.units = best_match['units']
            self.confidence = best_match['confidence']
            self.physical_length = best_match['physical_length']
            self.text_bbox = adjust_bbox_to_image_coords(best_match['bounding_box'], self.roi_bbox[0])
            self.scale_bar_bbox = adjust_bbox_to_image_coords(self.__detect_scale_bar(), self.roi_bbox[0])
            self.units_per_pixel = self.get_pixel_per_units()
            if self.scale_bar_bbox is not None:
                self.pixel_length = max(get_bbox_dimensions(self.scale_bar_bbox))
                self.units_per_pixel = self.physical_length / self.pixel_length

    def get_pixel_per_units(self):
        _, rc_mask = apply_rc_thresholding(grayscale_image(load_rgb_image(self.image_src)))
        return get_length_per_pixel(rc_mask, actual_length_micrometers=self.physical_length)


    def __detect_scale_bar(self):
        """
        Detects the scale bar in a microscopic image.
        
        Parameters:
            image_path (str): Path to the image file.
        
        Returns:
            list: Bounding box of the detected white bar in the format:
                [[np.int32(x1), np.int32(y1)],
                [np.int32(x2), np.int32(y2)],
                [np.int32(x3), np.int32(y3)],
                [np.int32(x4), np.int32(y4)]]
        """
        # Load the image
        image = cv2.imread(self.image_src)
        image = crop_from_bbox(image, self.roi_bbox)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to highlight the white bar
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on aspect ratio (long and thin bar)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0
            
            # Heuristic filter: must be a long and thin rectangle
            if aspect_ratio > 5 and w > 50 and h > 4:
                # Return bounding box in the required format
                return [[np.int32(x), np.int32(y)],
                        [np.int32(x + w), np.int32(y)],
                        [np.int32(x + w), np.int32(y + h)],
                        [np.int32(x), np.int32(y + h)]]
                
        # Canny Edge Detection fallback - this is needed for images with bright/white backgrounds
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(gray, 50, 150)  # Detect edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on aspect ratio (long and thin bar)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0
            
            # Heuristic filter: must be a long and thin rectangle
            if aspect_ratio > 5 and w > 50 and h > 4:
                # Return bounding box in the required format
                return [[np.int32(x), np.int32(y)],
                        [np.int32(x + w), np.int32(y)],
                        [np.int32(x + w), np.int32(y + h)],
                        [np.int32(x), np.int32(y + h)]]
        
        return None  # Return None if no bar is found
  
    def debug_display(self, crop_to_roi=False):
        """
        Loads an image, overlays the text and rectangle bounding boxes, and displays the image.
        """
        image = load_rgb_image(self.image_src)
        if image is None:
            print("Error: Unable to load image.")
            return
        
        if self.roi_bbox is not None:
            roi_bbox_arr = np.array(self.roi_bbox, dtype=np.int32)
            cv2.polylines(image, [roi_bbox_arr], isClosed=True, color=(0, 0, 255), thickness=2)  # Overlay the ROI bounding box
            cv2.putText(image, "ROI Region", tuple(roi_bbox_arr[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        if self.text_bbox is not None:
            text_bbox_arr = np.array(self.text_bbox, dtype=np.int32)
            cv2.polylines(image, [text_bbox_arr], isClosed=True, color=(0, 255, 0), thickness=2)  # Overlay the text bounding box
            cv2.putText(image, "Text", tuple(text_bbox_arr[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        if self.scale_bar_bbox is not None:
            scale_bar_bbox_arr = np.array(self.scale_bar_bbox, dtype=np.int32)
            cv2.polylines(image, [scale_bar_bbox_arr], isClosed=True, color=(255, 0, 0), thickness=2)  # Overlay the scale bar bounding box
            cv2.putText(image, "Scale Bar", tuple(scale_bar_bbox_arr[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        plt.figure(figsize=(10, 10))
        if crop_to_roi:
            image = crop_from_bbox(image, self.roi_bbox)
        plt.imshow(image)
        plt.show()
        

def adjust_bbox_to_image_coords(bbox, offset):
    """
    Shifts a bounding box from ROI-relative coordinates to full-image coordinates.

    Parameters:
        bbox (list): List of 4 [x, y] points (OCR-style).
        offset (tuple): (x_offset, y_offset) — the top-left corner of the ROI in full-image coords.

    Returns:
        list: Adjusted bounding box in full-image coordinates.
    """
    x_offset, y_offset = offset
    return [[x + x_offset, y + y_offset] for x, y in bbox]

        
def get_bottom_right_bbox(image, width_ratio=0.5, height_ratio=0.5):
    h, w = image.shape[:2]
    x1 = int(w * (1 - width_ratio))
    y1 = int(h * (1 - height_ratio))
    x2 = w
    y2 = h

    bbox = [
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ]
    return bbox

def crop_from_bbox(image, bbox):
    bbox_np = np.array(bbox, dtype=np.int32)

    # Compute axis-aligned bounding box (min/max x and y)
    x_min = max(int(np.min(bbox_np[:, 0])), 0)
    x_max = min(int(np.max(bbox_np[:, 0])), image.shape[1])
    y_min = max(int(np.min(bbox_np[:, 1])), 0)
    y_max = min(int(np.max(bbox_np[:, 1])), image.shape[0])

    return image[y_min:y_max, x_min:x_max]

def get_bbox_dimensions(bbox):
    """
    Given a bounding box defined by 4 points in the order:
    [top-left, top-right, bottom-right, bottom-left],
    returns the average width and height of the bounding box using NumPy.
    
    Parameters:
        bbox (list or np.ndarray): A list or array of four [x, y] coordinates.
        
    Returns:
        tuple: (width, height) as floats.
    """
    if bbox is None:
        return None
    bbox = np.array(bbox)
    if bbox.shape != (4, 2):
        raise ValueError("Bounding box must be a 4x2 array or list of points.")
    
    # Compute width as the average of the top and bottom edge lengths
    width_top = np.linalg.norm(bbox[1] - bbox[0])
    width_bottom = np.linalg.norm(bbox[2] - bbox[3])
    width = (width_top + width_bottom) / 2.0
    
    # Compute height as the average of the left and right edge lengths
    height_left = np.linalg.norm(bbox[3] - bbox[0])
    height_right = np.linalg.norm(bbox[2] - bbox[1])
    height = (height_left + height_right) / 2.0
    
    return width, height


# original code
def get_length_per_pixel(rc_mask, actual_length_micrometers=200):
    _, longest_line = get_distance_per_pixel_using_longest_contour(rc_mask)
    if longest_line:
        x1, y1, x2, y2 = longest_line
        length_in_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        length_per_pixel = actual_length_micrometers / length_in_pixels
        return length_per_pixel
    
def get_distance_per_pixel_using_longest_contour(rc_mask):
    # Detect all contours in the binary image using skimage
    contours = measure.find_contours(rc_mask, level=0.8)

    # Detect horizontal lines in contours
    horizontal_lines = detect_horizontal_lines(contours)

    # Combine horizontal lines that are close to each other
    combined_lines = combine_close_lines(horizontal_lines, pixel_tolerance=5)

    # Overlay combined horizontal lines on the binary image and highlight the longest in red
    overlay_image, longest_line = overlay_horizontal_lines(rc_mask, combined_lines)
    
    return overlay_image, longest_line

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