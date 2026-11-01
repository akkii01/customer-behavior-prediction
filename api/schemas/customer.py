from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class CustomerPredictionRequest(BaseModel):
    frequency: float = Field(..., example=5.0)
    total_items: float = Field(..., example=120.0)
    monetary_value: float = Field(..., example=1450.0)
    recency: float = Field(..., example=35.0)
    tenure_days: float = Field(..., example=240.0)
    avg_order_value: float = Field(..., example=290.0)
    avg_basket_size: float = Field(..., example=24.0)
    purchase_velocity: float = Field(..., example=0.62)
    avg_days_between: float = Field(..., example=42.0)
    std_days_between: float = Field(..., example=15.0)
    engagement_score: float = Field(..., example=0.78)


class CustomerPredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    predicted_risk_level: str
    predicted_future_clv: float
    top_churn_drivers: List[Dict[str, Any]]