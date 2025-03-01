from co_mof_image_utils import load_rgb_image, grayscale_image, apply_rc_thresholding
from co_mof import apply_morphological_closing 
from co_mof import process_and_overlay_if_few_contours, normalize_and_find_contours, filter_and_remove_white_region_contours


# bok_choy = BokChoy('image_path.jpg')

config = {
    "disaply_all_contour_debug": False,

}

class BokChoy:
    def __init__(self, image_src, use_overlapping=False):
        self.image_src = image_src
        self.image = load_rgb_image(self.image_src)
        self.use_overlapping = use_overlapping
        self.length_per_pixel = 0.444  # TODO replace this with scale bar object
        self.isolated_aspect_ratios = []
        self.isolated_areas = []

        if config["disaply_all_contour_debug"]:
            self.display_contours_debug()


    def get_contours(self):
        self.closed_image = apply_morphological_closing(apply_rc_thresholding(grayscale_image(self.image)))
        self.contours = normalize_and_find_contours(self.closed_image)

    def display_contours_debug(self):
        pass

    def get_aspect_ratios_and_area(self):
        process_and_overlay_if_few_contours(self.image,
                                            self.contours,
                                            self.length_per_pixel
)