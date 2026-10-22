from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.services.prediction import predict_attack


app = FastAPI(
    title="CyberSentry AI",
    description="Network intrusion detection API",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "CyberSentry AI API is running",
        "status": "healthy",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "random_forest"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return predict_attack(request.features)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc