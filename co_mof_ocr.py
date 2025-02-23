import easyocr
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt

from co_mof import load_rgb_image


class ScaleBarDetector:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.physical_length = None
        self.units = None
        self.pixel_length = None
        self.confidence = None
        self.text_bbox = None
        self.scale_bar_bbox = None
        self.units_per_pixel = None
        self.set_scale_bar_attributes()

    def set_scale_bar_attributes(self):
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
        reader = easyocr.Reader(['en'])
        results = reader.readtext(self.image_path)
        
        # Regex pattern: one or more digits (the numerical length), optional whitespace, followed by one or more letters (the units)
        pattern = r'\b(\d+)\s*([a-zA-Z]+)\b'
        
        best_match = None
        highest_confidence = 0
        
        for bbox, text, confidence in results:
            found = re.findall(pattern, text)
            if found:
                number_str, units = found[0]
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
            self.text_bbox = best_match['bounding_box']
            self.scale_bar_bbox = self.detect_scale_bar()
            if self.scale_bar_bbox is not None:
                scale_bar_length = max(get_bbox_dimensions(self.scale_bar_bbox))
                self.units_per_pixel = self.physical_length / scale_bar_length

    def detect_scale_bar(self):
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
        image = cv2.imread(self.image_path)
        
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
            if aspect_ratio > 5 and w > 50:
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
            if aspect_ratio > 5 and w > 50:
                # Return bounding box in the required format
                return [[np.int32(x), np.int32(y)],
                        [np.int32(x + w), np.int32(y)],
                        [np.int32(x + w), np.int32(y + h)],
                        [np.int32(x), np.int32(y + h)]]
        
        return None  # Return None if no bar is found
  
    def debug_display(self):
        """
        Loads an image, overlays the text and rectangle bounding boxes, and displays the image.
        """
        image = load_rgb_image(self.image_path)
        if image is None:
            print("Error: Unable to load image.")
            return
        
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
        plt.imshow(image)
        plt.show()

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
