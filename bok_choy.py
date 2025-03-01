from co_mof import process_and_overlay_if_few_contours


bok_choy = BokChoy('image_path.jpg')

class BokChoy:
    def __init__(self, image_src, use_overlapping=False):
        self.image_src = image_src
        self.image = None
        self.use_overlapping = use_overlapping
        self.length_per_pixel = 0.444  # TODO replace this with scale bar object
        self.isolated_aspect_ratios = []
        self.isolated_areas = []

    def get_aspect_ratios_and_area(self):
        process_and_overlay_if_few_contours()