import os
import pytest
import pandas as pd
from PIL import Image
import torch
from src.data.dataset import HAM10000Dataset, CLASS_TO_IDX
from src.data.transforms import eval_transform

def test_dataset_item(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Create a dummy image
    img_path = img_dir / "I1.jpg"
    img = Image.new("RGB", (300, 300), color="red")
    img.save(img_path)
    
    data = {
        'lesion_id': ['L1'],
        'image_id': ['I1'],
        'dx': ['mel'],
        'split': ['train']
    }
    df = pd.DataFrame(data)
    
    dataset = HAM10000Dataset(df, str(img_dir), transform=eval_transform)
    
    assert len(dataset) == 1
    
    image, label = dataset[0]
    
    assert label == CLASS_TO_IDX['mel']
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
