import os
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_TO_IDX = {"bkl": 0, "mel": 1, "nv": 2}

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
