from pydantic import BaseModel
from typing import List, Optional,Literal
from datetime import datetime




class ComplaintRequest(BaseModel):
    text: str
    image_description: Optional[str] = None
class ComplaintResponse(BaseModel):
    category: str
    subCategory: str
    severity: Literal["high","low","medium"]
    reason: str
    # timestamp: datetime