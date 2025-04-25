from pydantic import BaseModel
from typing import List, Optional

# Request model for recommendation input
class RecommendationRequest(BaseModel):
    user_id: int
    item_name: str
    top_n: Optional[int] = 10

# Individual recommendation item in the response
class RecommendationItem(BaseModel):
    Prod_ID: int
    Name: str
    Brand: str
    CF_Score: Optional[float] = 0.0
    CB_Score: Optional[float] = 0.0
    Hybrid_Score: float

# Full response model
class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
