
import csv
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction import predict_attack, EXPECTED_FEATURES

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="CyberSentry AI",
    description="Network intrusion detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "http://localhost:5173",
    "https://cybersentry-ai-uzz2.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/csv", response_model=BatchPredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB upload limit.",
        )

    try:
        text_content = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text_content))

        if not reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail="The CSV file is empty or has no header row.",
            )

        headers = [header.strip() for header in reader.fieldnames if header]
        missing_columns = sorted(set(EXPECTED_FEATURES) - set(headers))
        unexpected_columns = sorted(set(headers) - set(EXPECTED_FEATURES))

        if missing_columns or unexpected_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "CSV columns do not match the model's expected features.",
                    "missing_columns": missing_columns,
                    "unexpected_columns": unexpected_columns,
                },
            )

        results = []
        successful = 0
        failed = 0

        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                (key.strip() if key else key): value
                for key, value in raw_row.items()
            }

            if None in row or any(value is None for value in row.values()):
                failed += 1
                results.append({
                    "row": row_number,
                    "status": "error",
                    "error": "Row has an incorrect number of columns.",
                })
                continue

            features = {}
            try:
                for feature in EXPECTED_FEATURES:
                    value = row[feature]

                    if feature in {"proto", "service", "state"}:
                        features[feature] = value
                    else:
                        features[feature] = float(value)

                prediction = predict_attack(features)
                successful += 1
                results.append({
                    "row": row_number,
                    "status": "success",
                    **prediction,
                })

            except (ValueError, TypeError, KeyError) as exc:
                failed += 1
                results.append({
                    "row": row_number,
                    "status": "error",
                    "error": str(exc),
                })

        return {
            "total_records": successful + failed,
            "successful_predictions": successful,
            "failed_predictions": failed,
            "results": results,
        }

    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV must use UTF-8 encoding.",
        ) from exc
    except csv.Error as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not parse the CSV file.",
        ) from exc
    finally:
        await file.close()

