import json
import numpy as np
import pandas as pd

from co_mof_ocr import ScaleBarDetector
from co_mof_image_utils import load_rgb_image, grayscale_image, apply_rc_thresholding
from co_mof import apply_morphological_closing, display_rc_closing_results
from co_mof import normalize_and_find_contours, filter_and_remove_white_region_contours, display_contours
from co_mof import display_contours_on_original_color
from co_mof import classify_contours_by_convexity, display_classified_contours_on_original_color
from co_mof import calculate_blue_contour_metrics, plot_blue_contours
from co_mof import calculate_area_statistics_for_blue_contours, plot_area_histogram_for_blue_contours
from co_mof import filter_contours_by_area_and_edge, plot_filtered_blue_contours
from co_mof import remove_edge_touching_contours, plot_filtered_red_contours
from co_mof import calculate_contour_areas, plot_bounding_boxes_with_areas
from co_mof import calculate_summary_with_std, plot_summary_bar_chart


# Importing functions for overlapping crystals detection
from co_mof import analyze_isolated_aspect_ratios
from co_mof import detect_and_group_lines_per_contour, plot_grouped_lines_per_contour_on_original_image
from co_mof import analyze_red_contour_aspect_ratios, filter_valid_red_bounding_boxes, generate_yellow_bounding_boxes, plot_red_and_yellow_bounding_boxes
from co_mof import calculate_overlapping_area, plot_overlapping_area_and_aspect_ratios
from co_mof import calculate_unprocessed_contour_areas, plot_unprocessed_contour_areas
from co_mof import OC_calculate_summary_with_std, OC_plot_summary_bar_chart

class BokChoy:
    def __init__(self, image_src, use_overlapping=False, config_src="config/display_plots.json"):
        # Load configuration
        with open(config_src, 'r') as json_file:
            self.config = json.load(json_file)

        # Load image and preprocess
        self.image_src = image_src
        self.image = load_rgb_image(self.image_src)
        self.image_grey = grayscale_image(self.image)
        self.use_overlapping = use_overlapping
        
        # Detect Scale Bar
        self.length_per_pixel = self._detect_scale_bar(image_src)

        # Initialize attributes
        self._initialize_contour_attributes()
        if self.use_overlapping:
            self._initialize_overlapping_attributes()
            
        # Run Analysis Pipeline
        self._run_analysis()
            
    def _detect_scale_bar(self, image_src):
        """Detect scale bar and determine the length per pixel conversion."""
        detector = ScaleBarDetector(image_src)
        if detector.units_per_pixel and detector.units_per_pixel >= 0.1:
            return detector.units_per_pixel
        print("[WARNING] Scale bar detection failed. Using default scale: 0.4444 units per pixel.")
        return 0.4444

    def _initialize_contour_attributes(self):
        """Initialize attributes to prevent AttributeErrors."""
        self.contours, self.blue_contours, self.red_contours = [], [], []
        self.bounding_boxes_blue, self.blue_areas, self.blue_aspect_ratios = [], [], []
        self.filtered_blue_contours, self.isolated_aspect_ratios, self.isolated_areas = [], [], []
        self.filtered_red_contours, self.clusters_areas, self.bounding_boxes_red = [], [], []
        self.min_area, self.upper_limit = 0, 0
        
    def _initialize_overlapping_attributes(self):
        """Initialize attributes related to overlapping crystal detection."""
        self.overlapping_aspect_ratios, self.overlapping_areas = [], []
        self.processed_contours, self.unprocessed_contours = [], []
        self.bounding_boxes_yellow = []

        # Overlapping Crystal Debug Settings
        self.config.update({
            "OC_display_grouped_lines": False,
            "OC_display_detected_overlapping_crystals": False,
            "OC_display_detected_clusters": False
        })

    def _run_analysis(self):
        """Runs the main pipeline for contour detection and analysis."""
        self.get_contours()
        self.process_contours()
        self.collect_results()

        if self.config.get("display_all_contour_debug", False):
            self.display_contours_debug()


    # -----------------------------------------------
    # 1. Processing Pipeline
    # -----------------------------------------------

    def process_contours(self):
        """Determine the appropriate analysis pipeline based on the number of detected contours."""
        if len(self.contours) <= 2:
            print("[WARNING] Too few contours detected. Running minimal analysis...")
            self.too_few_detected_contours()
            
        else:
            self.contours_classification_and_filtering()
            self.isolated_crystals_analysis()
            self.clusters_analysis()
            
            if self.use_overlapping == True:
                self.overlapping_crystals_and_clusters_analysis()
            
        # Perform data analysis in both cases
        self.data_analysis()
    
    
    # -----------------------------------------------
    # 2. Contour Detection
    # -----------------------------------------------
    def get_contours(self):
        """Detect contours after preprocessing the image."""
        self.rc_thresh, self.rc_mask = apply_rc_thresholding(self.image_grey)
        
        self.closed_image = apply_morphological_closing(self.rc_mask)
        self.contours = normalize_and_find_contours(self.closed_image)
        self.contours = filter_and_remove_white_region_contours(self.closed_image, self.contours)
        
        if not self.contours:
            print("[WARNING] No contours found. Skipping further analysis.")
            return
  
    # New Code Block
        
    def contours_classification_and_filtering(self):
        """Classify contours as blue or red based on convexity."""
        if not self.contours:
            print("No contours to classify")
            
            self.blue_contours, self.red_contours = [], []
            return
        
        self.blue_contours, self.red_contours = classify_contours_by_convexity(self.image_grey.shape, self.contours)
        self.bounding_boxes_blue, self.blue_areas, self.blue_aspect_ratios = calculate_blue_contour_metrics(self.image_grey, self.blue_contours)
        self.x_area, self.y_area, self.peak_area, self.mean_area, self.std_area = calculate_area_statistics_for_blue_contours(self.blue_areas)
    
        # Adjust mean area based on heuristic thresholds
        self.mean_area = 500 if self.mean_area > 5000 else 200 if self.mean_area < 1000 else self.mean_area
    
        
    # -----------------------------------------------
    # 3. Filtering & Analysis
    # -----------------------------------------------
    
    def isolated_crystals_analysis(self):
        """Filter and analyze isolated crystals based on area thresholds."""
        if not self.blue_contours:
            print("[WARNING] No blue contours found.")
            return
        
        self.filtered_blue_contours, self.isolated_areas, self.isolated_aspect_ratios, self.filtered_red_contours, self.min_area, self.upper_limit = filter_contours_by_area_and_edge(
            self.blue_contours, self.red_contours, self.image_grey, self.mean_area, self.std_area, self.length_per_pixel
        )
        
        if self.use_overlapping:
            self.mean_isolated_ar, self.std_dev_isolated_ar = analyze_isolated_aspect_ratios(self.isolated_aspect_ratios)
        
    # New Code Block
        
    def clusters_analysis(self):
        """Analyze and filter red contours that represent clusters."""
        if not self.filtered_red_contours:
            print("[WARNING] No red contours available for clustering.")
            return
        
        self.filtered_red_contours = remove_edge_touching_contours(self.filtered_red_contours, self.image_grey)
        self.clusters_areas, self.bounding_boxes_red = calculate_contour_areas(self.filtered_red_contours, self.length_per_pixel)
        
    # New Code Block
    
    def too_few_detected_contours(self):
        """Minimal analysis when very few contours are detected."""
        if not self.contours:
            print("[WARNING] No contours available.")
            return
        
        self.bounding_boxes_blue, blue_areas, self.isolated_aspect_ratios = calculate_blue_contour_metrics(self.image_grey, self.contours)
        
        # Convert blue_areas to NumPy array before scaling
        self.isolated_areas = np.array([int(x) for x in blue_areas]) * self.length_per_pixel

        self.filtered_blue_contours = self.contours
        self.min_area = 1
        self.upper_limit = 10e8
        
    # New Code Block
        
    def overlapping_crystals_and_clusters_analysis(self):
        """Analyze overlapping crystal formations using aspect ratios and bounding boxes."""
        self.grouped_lines_per_contour = detect_and_group_lines_per_contour(self.image_grey, self.filtered_red_contours, group_threshold=10)
        self.bounding_boxes_data = analyze_red_contour_aspect_ratios(self.grouped_lines_per_contour, self.filtered_red_contours)
        
        self.bounding_boxes_red, self.overlapping_aspect_ratios, self.processed_contours = filter_valid_red_bounding_boxes(
        self.image_grey, self.bounding_boxes_data, min_AR=2, max_AR = self.mean_isolated_ar + 3 * self.std_dev_isolated_ar
        )
        self.bounding_boxes_yellow, self.unprocessed_contours = generate_yellow_bounding_boxes(self.image_grey, self.filtered_red_contours, self.processed_contours)

        self.overlapping_areas = calculate_overlapping_area(self.closed_image, self.bounding_boxes_red, self.length_per_pixel)
        self.clusters_areas, self.bounding_boxes_yellow = calculate_unprocessed_contour_areas(self.unprocessed_contours, self.length_per_pixel)
        
        
    # -----------------------------------------------
    # 4. Data Processing & Visualization
    # -----------------------------------------------
    
    def data_analysis(self):
        """Perform data analysis and generate summary tables with bar chart visualization."""

        if self.use_overlapping:
            # Calculate summary table with standard deviations (including overlapping crystals)
            summary_table_with_std = OC_calculate_summary_with_std(
                self.isolated_aspect_ratios, self.overlapping_aspect_ratios,
                self.isolated_areas, self.overlapping_areas, self.clusters_areas
            )
            
            # Ensure the plot displays the categories in the correct order
            summary_table_with_std = summary_table_with_std.iloc[::-1].reset_index(drop=True)
            self.results_summary_df = summary_table_with_std.copy()

        else:
            # Calculate summary table with standard deviations (without overlapping crystals)
            summary_table_with_std = calculate_summary_with_std(
                self.isolated_aspect_ratios, self.isolated_areas, self.clusters_areas
            )

            # Ensure the plot displays the categories in the correct order
            summary_table_with_std = summary_table_with_std.iloc[::-1].reset_index(drop=True)
            self.results_summary_df = summary_table_with_std.copy()

            
    def display_summary_chart(self):
        if self.use_overlapping:  # Generate summary bar chart for overlapping case
            summary = self.results_summary_df.copy()
            # Update category labels to have one word per line for better readability
            summary["Category"] = summary["Category"].replace({
                "Isolated Crystals": "Isolated\nCrystals",
                "Overlapping Crystals": "Overlapping\nCrystals",
                "Clusters": "Clusters"
            })
            OC_plot_summary_bar_chart(summary)
            
        else:  # Generate summary bar chart for non-overlapping case
            summary = self.results_summary_df.copy()
            # Update category labels to have one word per line for better readability
            summary["Category"] = summary["Category"].replace({
                "Isolated Crystals": "Isolated\nCrystals",
                "Clusters": "Clusters"
            })
            plot_summary_bar_chart(summary)
 
    def save_results_summary_to_csv(self, dst):
        self.results_summary_df.replace("-", None).to_csv(dst, index=False)
        print(f'Saved Bok Choy Summary results to {dst}')
        
    def collect_results(self):
        self.isolated_results_df = pd.DataFrame({'Category': ["Isolated Crystal" for i in range(len(self.isolated_areas))],
                                                 'Aspect Ratio': self.isolated_aspect_ratios,
                                                 'Area': self.isolated_areas})
        self.cluster_results_df = pd.DataFrame({'Category': ["Cluster" for i in range(len(self.clusters_areas))],
                                                'Area': self.clusters_areas})
        if not self.use_overlapping:
            self.results_df = pd.concat([self.isolated_results_df, self.cluster_results_df])
            return
        self.overlapping_results_df = pd.DataFrame({'Category': ["Overlapping" for i in range(len(self.overlapping_areas))],
                                                    'Aspect Ratio': self.overlapping_aspect_ratios,
                                                    'Area': self.overlapping_areas})
        self.results_df = pd.concat([self.isolated_results_df, self.cluster_results_df, self.overlapping_results_df])
    
    def save_results_to_csv(self, dst):
        self.results_df.to_csv(dst, index=False)
        print(f'Saved Bok Choy results to {dst}')
        
    def display_contours_debug(self):
        """Optional debug visualization for contours."""
    
        # Collect active debug settings
        active_debug_options = {key: value for key, value in self.config.items() if value is True}

        # If no debug options are enabled, skip visualization
        if not active_debug_options:
            print("[INFO] No debug visualizations enabled.")
            return
        
        print(f"[DEBUG] Active Visualization Configurations: {active_debug_options}")

        # General Contour Debugging
        debug_functions = {
            "display_rc_thresholding": lambda: display_rc_closing_results(self.image_grey, self.rc_thresh, self.rc_mask, self.closed_image),
            "display_detected_contours": lambda: display_contours(self.closed_image, self.contours),
            "display_detected_contours_on_original_image": lambda: display_contours_on_original_color(self.image, self.contours),
            "display_classified_contours": lambda: display_classified_contours_on_original_color(self.image, self.blue_contours, self.red_contours),
            "display_all_blue_contours": lambda: plot_blue_contours(self.image, self.bounding_boxes_blue),
            "display_area_histogram_for_blue_contours": lambda: plot_area_histogram_for_blue_contours(
                self.blue_areas, self.x_area, self.y_area, self.peak_area, self.mean_area, self.std_area
            ),
            "display_detected_isolated_crystals": lambda: plot_filtered_blue_contours(
                self.image, self.filtered_blue_contours, self.isolated_areas, self.isolated_aspect_ratios, self.min_area, self.upper_limit
            ),
            "display_detected_all_filtered_red_contours": lambda: plot_filtered_red_contours(self.image, self.filtered_red_contours, color="red"),
            "display_detected_clusters": lambda: plot_bounding_boxes_with_areas(self.image, self.bounding_boxes_red, self.clusters_areas, self.min_area, self.length_per_pixel)
        }

        # Overlapping Crystals Debugging (if enabled)
        overlapping_debug_functions = {
            "OC_display_grouped_lines": lambda: plot_grouped_lines_per_contour_on_original_image(self.image, self.grouped_lines_per_contour),
            "OC_display_detected_overlapping_crystals": lambda: plot_overlapping_area_and_aspect_ratios(self.image, self.bounding_boxes_red, self.overlapping_areas, self.overlapping_aspect_ratios),
            "OC_display_detected_clusters": lambda: plot_unprocessed_contour_areas(self.image, self.bounding_boxes_yellow, self.clusters_areas)
        }

        # Execute general contour debugging functions
        for key, func in debug_functions.items():
            if self.config.get(key, False):
                func()

        # Execute overlapping debugging functions if enabled
        if self.use_overlapping:
            for key, func in overlapping_debug_functions.items():
                if self.config.get(key, False):
                    func()

            