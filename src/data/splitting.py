import pandas as pd

def load_splits(csv_path: str):
    """
    Loads the pre-calculated splits from the selected_images.csv.
    Ensures that there is no leakage of lesion_ids across splits.
    """
    df = pd.read_csv(csv_path)
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    return train_df, val_df, test_df

def verify_no_leakage(train_df, val_df, test_df):
    """
    Verifies that no lesion_id appears in more than one split.
    Raises ValueError if leakage is detected.
    """
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    test_lesions = set(test_df['lesion_id'])
    
    if train_lesions.intersection(val_lesions):
        raise ValueError("Leakage detected between train and val splits.")
    if train_lesions.intersection(test_lesions):
        raise ValueError("Leakage detected between train and test splits.")
    if val_lesions.intersection(test_lesions):
        raise ValueError("Leakage detected between val and test splits.")
        
    return True
