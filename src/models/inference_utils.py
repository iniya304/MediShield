import torch
import torch.nn.functional as F

def get_prediction_bundle(model, image_tensor, device, feature_layer_name="avgpool"):
    """
    Returns: probs (np.array), pred_class (int), confidence (float),
             deep_features (np.array) — extracted via forward hook.
    """
    model.eval()
    features = {}
    def hook(module, inp, out):
        features["feat"] = out.detach().flatten(1)

    layer = dict(model.named_modules())[feature_layer_name]
    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    handle.remove()
    pred_class = int(probs.argmax())
    confidence = float(probs.max())
    deep_feat = features["feat"].cpu().numpy()[0]
    return probs, pred_class, confidence, deep_feat
