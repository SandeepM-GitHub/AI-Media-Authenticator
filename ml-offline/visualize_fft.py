import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.image_loader import ImageLoader

def visualize_fft(image_path):
    loader = ImageLoader()
    image = loader.load(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1,2,1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('FFT Magnitude Spectrum')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    visualize_fft("data/ai/lion.jpg")