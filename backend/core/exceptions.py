# core/exceptions.py
"""
커스텀 예외
"""
from fastapi import HTTPException


class RAGNotAvailableError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=503,
            detail="RAG system not available"
        )


class DatabaseNotAvailableError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=503,
            detail="Database not available"
        )


class RecipeNotFoundError(HTTPException):
    def __init__(self, recipe_id: int):
        super().__init__(
            status_code=404,
            detail=f"Recipe {recipe_id} not found"
        )


class SessionNotFoundError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(
            status_code=404,
            detail=f"Session {session_id} not found"
        )