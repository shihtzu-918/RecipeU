# backend/features/user/schemas.py
"""
User 관련 스키마 (MySQL 기반)
"""
from pydantic import BaseModel
from typing import List, Optional


class UserProfileResponse(BaseModel):
    id: Optional[int] = None
    name: str
    email: Optional[str] = None
    allergies: List[str] = []
    dislikes: List[str] = []


class FamilyMemberInfo(BaseModel):
    id: int
    relationship: str = ""
    allergies: List[str] = []
    dislikes: List[str] = []


class FamilyInfoResponse(BaseModel):
    family_members: List[FamilyMemberInfo] = []


class AllConstraintsResponse(BaseModel):
    allergies: List[str] = []
    dislikes: List[str] = []
    utensils: List[str] = []