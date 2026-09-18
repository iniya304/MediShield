import os
import google.generativeai as genai

def generate_clinical_summary(pred_class, conf, rel, q_score, decision):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key not configured. Cannot generate text summary."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    You are an AI medical assistant part of the MediShield skin lesion classification pipeline.
    The technical pipeline has produced the following results for a dermoscopy image:
    - Predicted Class: {pred_class}
    - Model Confidence (MSP): {conf:.2f}
    - Safety Layer Reliability Score: {rel:.2f}
    - Explainability Quality (Q-Score): {q_score:.2f}
    - Final Pipeline Decision: {decision}
    
    Write a short, professional, 3-sentence summary meant for a doctor explaining what these metrics mean 
    for this specific image, and whether they should trust the system's prediction. Keep it concise.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"
