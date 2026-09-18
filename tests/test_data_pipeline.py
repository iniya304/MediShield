import pytest
import pandas as pd
import numpy as np
import torch
from src.utils.seed import set_seed
from src.data.splitting import load_splits, verify_no_leakage

def test_set_seed():
    set_seed(42)
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()
    
    set_seed(42)
    np_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()
    
    assert np_val1 == np_val2
    assert torch_val1 == torch_val2

def test_load_and_verify_splits(tmp_path):
    # Create a dummy csv
    csv_file = tmp_path / "dummy_selected_images.csv"
    data = {
        'lesion_id': ['L1', 'L1', 'L2', 'L3', 'L4'],
        'image_id': ['I1', 'I2', 'I3', 'I4', 'I5'],
        'dx': ['mel', 'mel', 'nv', 'bkl', 'nv'],
        'split': ['train', 'train', 'val', 'test', 'test']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    train_df, val_df, test_df = load_splits(str(csv_file))
    
    assert len(train_df) == 2
    assert len(val_df) == 1
    assert len(test_df) == 2
    
    assert verify_no_leakage(train_df, val_df, test_df) == True

def test_leakage_detection(tmp_path):
    csv_file = tmp_path / "leak_selected_images.csv"
    data = {
        'lesion_id': ['L1', 'L1', 'L2'],
        'image_id': ['I1', 'I2', 'I3'],
        'dx': ['mel', 'mel', 'nv'],
        'split': ['train', 'val', 'test']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    train_df, val_df, test_df = load_splits(str(csv_file))
    with pytest.raises(ValueError, match="Leakage detected between train and val splits."):
        verify_no_leakage(train_df, val_df, test_df)
