from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionResponse
from .dependencies import get_cnn_model, get_xgb_model, get_config
from .services.pipeline import process_image

app = FastAPI(title="MediShield API", description="Skin Lesion Classification & Safety Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MediShield API is running."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    cnn_model = Depends(get_cnn_model),
    xgb_model = Depends(get_xgb_model),
    config = Depends(get_config)
):
    image_bytes = await file.read()
    results = process_image(image_bytes, cnn_model, xgb_model, config)
    return results
