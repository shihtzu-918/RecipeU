# features/chat/schemas.py
"""
Chat 관련 Pydantic 모델
"""
from typing import TypedDict, Annotated, List, Dict
import operator


class ChatAgentState(TypedDict):
    """Chat Agent 상태"""
    messages: Annotated[list, operator.add]
    user_constraints: dict
    search_query: str
    retrieved_recipes: list
    filtered_recipes: list
    selected_recipe: dict
    response: str
    step: str


class ChatMessage(TypedDict):
    """채팅 메시지"""
    role: str
    content: str
    recipe_info: Dict | None
    timestamp: str