from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .models import (
    OrderBookSequence,
    PredictionResponse,
    StatusResponse,
    HealthResponse
)
from .utils import get_model, get_pipeline, model_loader, app_start_time

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/", response_model=StatusResponse)
def read_root():
    """Get API status"""
    uptime = str(datetime.now() - app_start_time)

    response = {
        "status": "running",
        "uptime": uptime,
        "model_loaded": model_loader.model is not None
    }

    if model_loader.load_time:
        response["model_load_time"] = model_loader.load_time.isoformat()

    return response

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health"""
    if model_loader.model is None:
        return HealthResponse(status=503, message="Model not loaded")
    return HealthResponse(status=200, message="OK")

@router.post("/predict", response_model=PredictionResponse)
def predict(sequence: OrderBookSequence,
            model=Depends(get_model),
            pipeline=Depends(get_pipeline)):
    """Make prediction for a new observation"""
    start_time = datetime.now()

    try:
        # Convert pydantic model to DataFrame
        df = pd.DataFrame([event.dict() for event in sequence.events])

        # Add observation ID column required by the pipeline
        df['obs_id'] = 'new_observation'

        # Process using pipeline
        reshaper = pipeline.named_steps['reshaper']
        vectorizer = pipeline.named_steps['vectorizer']

        # Transform sequence
        sequence_dict = reshaper.transform({'new_observation': df})
        X_tensor, _ = vectorizer.transform(sequence_dict)

        # Make prediction
        probabilities = model.model.predict(X_tensor)[0]
        prediction = np.argmax(probabilities)
        prediction_prob = float(probabilities[prediction])

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "prediction": int(prediction),
            "prediction_probability": prediction_prob,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/reload-model")
def reload_model():
    """Force reload of the model"""
    success = model_loader.load_model()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload model")
    return {"message": "Model reloaded successfully", "load_time": model_loader.load_time.isoformat()}