import torch
from src.models.efficientnet import build_efficientnet
from src.reliability.xgb_detector import XGBoostReliabilityDetector
import yaml
import os

_MODEL = None
_XGB_MODEL = None
_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        if os.path.exists("config/config.yaml"):
            with open("config/config.yaml", 'r') as f:
                _CONFIG = yaml.safe_load(f)
        else:
            _CONFIG = {"safety": {"reliability_threshold": 0.8, "quality_threshold": 0.7}}
    return _CONFIG

def get_cnn_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_efficientnet(num_classes=3, pretrained=False)
        model_path = "models/efficientnet_b0.pth"
        if os.path.exists(model_path):
            _MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
        _MODEL.eval()
    return _MODEL

def get_xgb_model():
    global _XGB_MODEL
    if _XGB_MODEL is None:
        _XGB_MODEL = XGBoostReliabilityDetector()
        xgb_path = "models/reliability_xgb.json"
        if os.path.exists(xgb_path):
            _XGB_MODEL.load(xgb_path)
    return _XGB_MODEL
