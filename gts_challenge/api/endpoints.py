from fastapi import APIRouter, HTTPException, Depends, Body
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time  # For timing processing
from enum import Enum  # Import Enum

from .models import (
    OrderBookSequence,
    PredictionRequest,  # Use the new request model
    PredictionResponse,
    ExplanationRequest,  # New request model for explanation
    ExplanationResponse,  # New response model for explanation
    StatusResponse,
    HealthResponse
)
from .utils import model_loader, get_uptime, app_start_time  # Import get_uptime
from gts_challenge.order_book.factory import api_predict_workflow, api_explain_workflow

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# --- Define Enum for Model Types ---
class ModelType(str, Enum):
    gru = "gru"
    gb = "gb"


@router.get("/", response_model=StatusResponse, tags=["Status"])  # <-- ADDED TAG
def read_root():
    """Get API status"""
    uptime = get_uptime()
    # Ensure all keys expected by StatusResponse are present
    return StatusResponse(
        status="API is running",
        start_time=app_start_time.isoformat(),  # Assuming app_start_time is available
        uptime=uptime,
        loaded_models=list(model_loader.loaded_models.keys()),  # List of keys
        model_load_times={k: v.isoformat() if v else None for k, v in model_loader.load_times.items()},
        loaded_pipelines=list(model_loader.loaded_pipelines.keys()),  # List of keys
        pipeline_load_times={k: v.isoformat() if v else None for k, v in model_loader.pipeline_load_times.items()}
    )


@router.get("/health", response_model=HealthResponse, tags=["Status"])  # <-- ADDED TAG
def health_check():
    """Check API health"""
    # Basic check: API is up
    # More sophisticated checks could be added here (e.g., model loading status)
    models_ok = bool(model_loader.get_model('gru')) and bool(model_loader.get_model('gb'))
    pipelines_ok = bool(model_loader.get_pipeline('gru')) and bool(model_loader.get_pipeline('gb'))
    status = "pass" if models_ok and pipelines_ok else "fail"
    details = {
        "gru_model_loaded": bool(model_loader.get_model('gru')),
        "gb_model_loaded": bool(model_loader.get_model('gb')),
        "gru_pipeline_loaded": bool(model_loader.get_pipeline('gru')),
        "gb_pipeline_loaded": bool(model_loader.get_pipeline('gb')),
    }
    # Return structure matching HealthResponse
    return HealthResponse(status=status, details=details)


# --- Use ModelType Enum for path parameter ---
@router.post("/reload-pipeline/{model_type}", tags=["Management"])  # <-- ADDED TAG
async def reload_pipeline(model_type: ModelType):  # Use the Enum here
    """Force reload of a specific pipeline (gru or gb)"""
    logger.info(f"Attempting to reload pipeline for model type: {model_type.value}")
    # Invalidate cache by removing from loader dict
    model_loader._pipelines.pop(model_type.value, None)
    model_loader._pipeline_load_times[model_type.value] = None
    # Attempt to reload immediately
    success = model_loader.load_pipeline(model_type.value)
    if success:
        return {"message": f"Pipeline for '{model_type.value}' reloaded successfully."}
    else:
        # Keep 500 if loading itself fails after validation passes
        raise HTTPException(
            status_code=500, detail=f"Failed to reload pipeline for '{model_type.value}'. Check logs.")


# --- Use ModelType Enum for path parameter ---
@router.post("/reload-model/{model_type}", tags=["Management"])  # <-- ADDED TAG
async def reload_model(model_type: ModelType):  # Use the Enum here
    """Force reload of a specific model (gru or gb)"""
    logger.info(f"Attempting to reload model: {model_type.value}")
    # Invalidate cache by removing from loader dict
    model_loader._models.pop(model_type.value, None)
    model_loader._load_times[model_type.value] = None
    # Attempt to reload immediately
    success = model_loader.load_model(model_type.value)
    if success:
        return {"message": f"Model '{model_type.value}' reloaded successfully."}
    else:
        # Keep 500 if loading itself fails after validation passes
        raise HTTPException(
            status_code=500, detail=f"Failed to reload model '{model_type.value}'. Check logs.")

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    start_time_proc = time.perf_counter()
    model_type = request.model_type
    sequence = request.sequence

    if model_type not in [m.value for m in ModelType]:
        raise HTTPException(
            status_code=422, detail=f"Invalid model_type '{model_type}' in request body. Must be 'gru' or 'gb'.")

    # --- Get Model and Pipeline Objects ---
    try:
        # Use the loader to get the actual loaded objects
        model = model_loader.get_model(model_type)
        pipeline = model_loader.get_pipeline(model_type)
        # REMOVE: workflow = model_loader.get_workflow(model_type) # No longer needed here

        # Check if loading succeeded (get_model/get_pipeline now raise exceptions on failure)
        # The checks below might be redundant if get_model/get_pipeline always raise
        if not model or not pipeline:
            # This path might not be reachable if get_model/get_pipeline raise exceptions
            logger.error(f"Model or pipeline object is None after loading for {model_type}")
            raise HTTPException(
                status_code=503, detail=f"Model or pipeline could not be retrieved for {model_type}.")

    except HTTPException as e:
        # Re-raise HTTPExceptions raised by get_model/get_pipeline
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error getting model/pipeline for {model_type}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error loading resources for {model_type}.")

    try:
        # --- Preprocess data ---
        df = pd.DataFrame([event.model_dump() for event in sequence.events])
        df['obs_id'] = 0 # Add obs_id column if needed by the workflow/pipeline

        # --- Call the API workflow function from factory ---
        # Pass model_type, and the loaded model and pipeline objects
        predictions_df = api_predict_workflow(
            model_type=model_type,
            model=model,
            pipeline=pipeline,
            X_df=df
        )

        # --- Process results ---
        if isinstance(predictions_df, pd.DataFrame) and 'prediction' in predictions_df.columns:
            # Assuming the workflow returns a DataFrame with a 'prediction' column
            prediction = int(predictions_df['prediction'].iloc[0])
        else:
            # Log error if the format is unexpected
            logger.error(f"Unexpected output format from api_predict_workflow for {model_type}: {type(predictions_df)}")
            raise HTTPException(status_code=500, detail="Internal error processing prediction output.")

        end_time_proc = time.perf_counter()
        processing_time_ms = int((end_time_proc - start_time_proc) * 1000)

        return PredictionResponse(
            model_type=model_type,
            prediction=prediction,
            processing_time_ms=processing_time_ms
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error making prediction with {model_type}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error with {model_type}: {str(e)}")


@router.post("/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_prediction(request: ExplanationRequest):
    start_time_total = time.perf_counter()
    logger.info(f"Received explanation request for model type: {request.model_type}")
    model_type = request.model_type

    if model_type not in [m.value for m in ModelType]:
         raise HTTPException(status_code=400, detail="Invalid model_type. Use 'gru' or 'gb'.")

    try:
        # --- Get Model and Pipeline Objects ---
        model = model_loader.get_model(model_type)
        pipeline = model_loader.get_pipeline(model_type)
        # REMOVE: workflow = model_loader.get_workflow(model_type) # No longer needed here

        if not model or not pipeline:
             # This path might not be reachable if get_model/get_pipeline raise exceptions
             logger.error(f"Model or pipeline object is None after loading for {model_type}")
             raise HTTPException(
                 status_code=503, detail=f"Model or pipeline could not be retrieved for {model_type}.")

    except HTTPException as e:
         raise e
    except Exception as e:
         logger.exception(f"Unexpected error getting model/pipeline for explanation with {model_type}: {str(e)}")
         raise HTTPException(
             status_code=500, detail=f"Internal error loading resources for explanation with {model_type}.")

    try:
        # --- Preprocess data ---
        sequence_data = [event.model_dump() for event in request.sequence.events]
        df_sequence = pd.DataFrame(sequence_data)
        df_sequence['obs_id'] = 0 # Add obs_id column if needed

        # --- Call the API workflow function from factory ---
        explanation_results = api_explain_workflow(
            model_type=model_type,
            model=model,
            pipeline=pipeline,
            X_df=df_sequence # Ensure arg name matches function definition
        )

        # --- Process results ---
        feature_importance = []
        # Adjust based on the actual structure returned by api_explain_workflow
        if isinstance(explanation_results, dict) and 'feature_importances' in explanation_results:
            # Example: Convert dict to list of dicts
            importances_dict = explanation_results['feature_importances']
            if isinstance(importances_dict, dict):
                 feature_importance = [{"feature": f, "importance": float(v)} for f, v in importances_dict.items()]
            else:
                 # Handle other possible formats (e.g., list of tuples)
                 logger.warning(f"Received feature importances in unexpected format: {type(importances_dict)}")
                 # Attempt to adapt or raise error
                 raise HTTPException(status_code=500, detail="Internal error processing feature importance format.")

        else:
            logger.error(f"Unexpected output format from api_explain_workflow for {model_type}: {type(explanation_results)}")
            raise HTTPException(status_code=500, detail="Internal error processing explanation output.")

        end_time_total = time.perf_counter()
        processing_time_ms = int((end_time_total - start_time_total) * 1000)

        # Assuming explanation doesn't recalculate prediction
        prediction_value = explanation_results.get('prediction', 0) # Get prediction if returned, else default

        return ExplanationResponse(
            model_type=model_type,
            prediction=int(prediction_value), # Ensure prediction is int
            feature_importance=feature_importance,
            processing_time_ms=processing_time_ms
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error during explanation with {model_type}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Explanation error with {model_type}: {str(e)}")