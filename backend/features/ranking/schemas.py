# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# Pydantic 모델
class Ingredient(BaseModel):
    name: str
    desc: Optional[str] = None
    amount: Optional[str] = None
    category: Optional[str] = None


class RecipePreview(BaseModel):
    recipe_id: str
    title: str
    author: str
    image: str


class RankingResponse(BaseModel):
    date_kst: str
    recipes: List[RecipePreview]
    total_count: int


class RecipeDetail(BaseModel):
    recipe_id: str
    title: str
    author: Optional[str] = None
    image: str
    intro: Optional[str] = None
    portion: Optional[str] = None
    cook_time: Optional[str] = None
    level: Optional[str] = None
    detail_url: Optional[str] = None
    ingredients: List[Ingredient]
    steps: List[str]
    registered_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}