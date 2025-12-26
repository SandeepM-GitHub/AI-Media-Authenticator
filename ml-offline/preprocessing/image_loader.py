from PIL import Image
import numpy as np

class ImageLoader:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def load(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image_array = np.array(image) / 255.0
        return image_array