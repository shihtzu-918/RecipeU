# features/cooking/schemas.py
"""
Cooking 관련 Pydantic 모델
"""
from typing import TypedDict, Annotated, Literal
import operator


class CookingAgentState(TypedDict):
    """Cooking Agent 상태"""
    recipe: dict
    current_step: int
    total_steps: int
    history: Annotated[list, operator.add]
    user_input: str
    intent: str
    response: str
    audio_path: str
    next_action: Literal["next", "prev", "substitute", "failure", "continue", "end"]