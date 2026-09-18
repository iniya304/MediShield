import torch
from src.models.efficientnet import build_efficientnet
from src.models.resnet import build_resnet18
from src.models.inference_utils import get_prediction_bundle

def test_efficientnet_output_shape():
    model = build_efficientnet(num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 3)

def test_resnet_output_shape():
    model = build_resnet18(num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 3)

def test_inference_utils():
    # Use resnet for testing since it's smaller/faster, avgpool is named 'avgpool'
    model = build_resnet18(num_classes=3, pretrained=False)
    dummy_input = torch.randn(3, 224, 224)
    
    probs, pred_class, conf, feat = get_prediction_bundle(
        model, dummy_input, device='cpu', feature_layer_name='avgpool'
    )
    
    assert probs.shape == (3,)
    assert isinstance(pred_class, int)
    assert 0 <= pred_class < 3
    assert isinstance(conf, float)
    assert 0 <= conf <= 1.0
    assert len(feat.shape) == 1 # 512 for resnet18
