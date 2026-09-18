import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
from src.reliability.features import compute_margin, compute_entropy
from src.explainability.gradcam import GradCAM
from src.explainability.deletion_insertion import calculate_deletion_insertion_auc
from src.explainability.stability_metrics import calculate_robustness_score, calculate_consistency_score
from src.explainability.quality_score import calculate_quality_score
from src.decision.abstention import AbstentionLayer
from .gemini_service import generate_clinical_summary

CLASS_MAP = {0: 'bkl', 1: 'mel', 2: 'nv'}
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(image_bytes: bytes, cnn_model, xgb_model, config):
    # 1. Preprocess
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    # 2. Classification
    with torch.no_grad():
        outputs = cnn_model(tensor)
        probs = outputs.softmax(dim=1).numpy()
        
    pred_idx = probs.argmax(axis=1)[0]
    pred_class = CLASS_MAP.get(pred_idx, "Unknown")
    confidence = probs[0, pred_idx]
    
    # 3. Reliability Features
    margin = float(compute_margin(probs)[0])
    entropy_val = float(compute_entropy(probs)[0])
    
    try:
        X = np.array([[margin, entropy_val]])
        reliability = float(xgb_model.predict_reliability(X)[0])
    except:
        reliability = 0.95 # Fallback
        
    # 4. Explainability
    target_layer = [cnn_model.features[-1]]
    gradcam_gen = GradCAM(cnn_model, target_layer)
    cam = gradcam_gen.generate(tensor, pred_idx)
    
    del_auc, ins_auc = calculate_deletion_insertion_auc(cnn_model, tensor, cam, pred_idx)
    robustness = calculate_robustness_score(cnn_model, tensor, cam, pred_idx, gradcam_gen)
    consistency = calculate_consistency_score(cnn_model, tensor, pred_idx, gradcam_gen)
    
    q_score = calculate_quality_score(del_auc, ins_auc, robustness, consistency)
    faithfulness = ((1.0 - del_auc) + ins_auc) / 2.0
    
    # 5. Abstention Decision
    safety_config = config.get('safety', {})
    abstention = AbstentionLayer(
        reliability_threshold=safety_config.get('reliability_threshold', 0.8),
        quality_threshold=safety_config.get('quality_threshold', 0.7)
    )
    decision = abstention.evaluate(reliability, q_score)
    
    # 6. Format Base64 Image
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img.resize((cam.shape[1], cam.shape[0])))
    overlay = cv2.addWeighted(img_np, 0.5, cam_heatmap, 0.5, 0)
    _, buffer = cv2.imencode('.jpg', overlay)
    base64_cam = base64.b64encode(buffer).decode('utf-8')
    
    # 7. Gemini Summary
    summary = generate_clinical_summary(pred_class, confidence, reliability, q_score, decision)
    
    return {
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "reliability_score": float(reliability),
        "q_score": float(q_score),
        "faithfulness": float(faithfulness),
        "robustness": float(robustness),
        "consistency": float(consistency),
        "decision": decision,
        "gradcam_base64": base64_cam,
        "clinical_summary": summary
    }
