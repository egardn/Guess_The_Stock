from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from datetime import datetime

class OrderBookEvent(BaseModel):
    """Single order book event data"""
    venue: str
    action: str
    trade: bool
    bid: float
    ask: float
    price: float
    bid_size: float
    ask_size: float
    flux: float

class OrderBookSequence(BaseModel):
    """Sequence of order book events for prediction"""
    events: List[OrderBookEvent] = Field(
        ..., 
        min_items=100, 
        max_items=100,
        description="Sequence of 100 order book events"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int
    prediction_probability: float
    processing_time_ms: float
    timestamp: str

class StatusResponse(BaseModel):
    """API status response model"""
    status: str
    uptime: str
    model_loaded: bool
    model_load_time: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: int
    message: str