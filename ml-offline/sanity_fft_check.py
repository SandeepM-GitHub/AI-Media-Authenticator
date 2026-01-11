import numpy as np
from preprocessing.fft_features import FFTFeatureExtractor
import cv2
import glob

fft = FFTFeatureExtractor()

def stats(images):
    feats = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        feats.append(fft.extract(img))
    feats = np.stack(feats)
    return feats.mean(), feats.std()

real_imgs = glob.glob("data/real/butterfly.jpg")[:50]
ai_imgs   = glob.glob("data/ai/yogi.jpg")[:50]

print("REAL FFT:", stats(real_imgs))
print("AI FFT:", stats(ai_imgs))
