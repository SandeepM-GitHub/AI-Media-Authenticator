import os
import torch
from torch.utils.data import Dataset
from preprocessing.image_loader import ImageLoader

class ImageDataset(Dataset):
    def __init__(self, base_path):
        self.samples = []
        self.loader = ImageLoader()

        for label, folder in enumerate(["real", "ai"]):
            folder_path = os.path.join(base_path, folder)
            for file in os.listdir(folder_path):
                self.samples.append(
                    (os.path.join(folder_path, file), label)
                )
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader.load(path)
        image = torch.tensor(image).permute(2, 0, 1).float()
        return image, label
        