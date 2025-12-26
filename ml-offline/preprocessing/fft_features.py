import numpy as np
import cv2

class FFTFeatureExtractor:
    """Extracts FFT-based features from images."""

    def __init__(self, image_size=224):
        self.image_size = image_size

    def extract(self, image):
        """
        image: numpy array of shape (H, W, C), values in [0, 1]
        returns: frequecy feature vector
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        # Normalize
        magnitude = magnitude / np.max(magnitude)
        magnitude = cv2.resize(magnitude, (32, 32))

        # Reduce to compact feature
        flat = magnitude.flatten()
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)
        return flat