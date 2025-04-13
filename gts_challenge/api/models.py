from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Optional, Any
from datetime import datetime

class OrderBookEvent(BaseModel):
    """Single order book event data"""
    venue: int
    order_id: int # <-- Add order_id
    action: str
    side: str # <-- Add side
    trade: bool
    bid: float # Removed gt=0 constraint
    ask: float # Removed gt=0 constraint
    price: float # Removed ge=0 constraint
    bid_size: float = Field(..., gt=0) # Keep positivity for size
    ask_size: float = Field(..., gt=0) # Keep positivity for size
    flux: float

    @validator('action')
    def action_must_be_valid(cls, v):
        # Example: Add validation for known action types if available
        valid_actions = {'A', 'D', 'U'} # Based on README
        if v not in valid_actions:
            raise ValueError(f'Invalid action type: {v}. Must be one of {valid_actions}')
        return v

    @validator('side') # <-- Add validator for side
    def side_must_be_valid(cls, v):
        valid_sides = {'A', 'B'} # Based on README
        if v not in valid_sides:
            raise ValueError(f'Invalid side: {v}. Must be one of {valid_sides}')
        return v

    @validator('venue')
    def venue_must_be_valid(cls, v):
        # Example: Add validation for known venue types if available (e.g., range)
        # Based on README, venue is encoded as integer. Add range check if known.
        # You might want to add a check like:
        # if not (0 <= v <= 14): # Assuming 15 venues means IDs 0-14
        #     raise ValueError(f'Invalid venue ID: {v}. Must be between 0 and 14.')
        return v

class OrderBookSequence(BaseModel):
    """Sequence of order book events for prediction"""
    events: List[OrderBookEvent] = Field(..., min_items=100, max_items=100) # Ensure exactly 100 events

    @validator('events')
    def check_sequence_length(cls, v):
        if len(v) != 100:
            raise ValueError('Sequence must contain exactly 100 events')
        return v

class PredictionRequest(BaseModel):
    """Request model for prediction, specifying model type"""
    model_type: str # Keep as string, validated in endpoint ('gru' or 'gb')
    sequence: OrderBookSequence

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_type: str
    prediction: int
    processing_time_ms: int

class ExplanationRequest(BaseModel):
    """Request model for explanation, specifying model type"""
    model_type: str # Keep as string, validated in endpoint ('gru' or 'gb')
    sequence: OrderBookSequence
    
class ExplanationResponse(BaseModel):
    """Response model for explanations"""
    model_type: str
    prediction: int
    feature_importance: List[Dict[str, Any]] # List of {'feature': name, 'importance': value}
    processing_time_ms: int

class StatusResponse(BaseModel):
    """API status response model"""
    status: str
    start_time: str
    uptime: str
    loaded_models: List[str]
    model_load_times: Dict[str, Optional[str]]
    loaded_pipelines: List[str]
    pipeline_load_times: Dict[str, Optional[str]]

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str # 'pass' or 'fail'
    details: Dict[str, Any]