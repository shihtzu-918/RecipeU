# backend/features/recipe/schemas.py
"""
Recipe API 스키마
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class RecipeGenerateRequest(BaseModel):
    chat_history: List[Dict[str, str]] 
    member_info: Optional[Dict[str, Any]] = None

class RecipeResponse(BaseModel):
    recipe: Dict[str, Any]
    recipe_id: Optional[int] = None
    user_id: Optional[str] = None
    title: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

class RecentRecipeResponse(BaseModel):
    id: int
    title: str
    created_at: str