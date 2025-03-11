from fastapi import HTTPException
from datetime import datetime
import logging
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to model and pipeline files
MODEL_PATH = os.getenv("MODEL_PATH", "../order_book_model.pkl")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "../order_book_pipeline.pkl")

# Track API start time
app_start_time = datetime.now()

class ModelLoader:
    def __init__(self):
        self._model = None
        self._pipeline = None
        self._load_time = None

    def load_model(self):
        """Load model and pipeline from files"""
        try:
            with open(MODEL_PATH, 'rb') as f:
                self._model = pickle.load(f)

            with open(PIPELINE_PATH, 'rb') as f:
                self._pipeline = pickle.load(f)

            self._load_time = datetime.now()
            logger.info(f"Model loaded successfully at {self._load_time}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    @property
    def model(self):
        if not self._model:
            self.load_model()
        return self._model

    @property
    def pipeline(self):
        if not self._pipeline:
            self.load_model()
        return self._pipeline

    @property
    def load_time(self):
        return self._load_time

# Initialize model loader
model_loader = ModelLoader()

# Dependency to ensure model is loaded
def get_model():
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_loader.model

def get_pipeline():
    if model_loader.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return model_loader.pipeline