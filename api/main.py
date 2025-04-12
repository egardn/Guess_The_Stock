from fastapi import FastAPI
from .endpoints import router
from .utils import model_loader
import logging

# Configure logging
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="Order Book Prediction API",
    description="API for order book sequence classification",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = model_loader.load_model()
    if not success:
        logger.warning("Failed to load model at startup. Will attempt to load on first request.")

# Include router
app.include_router(router)