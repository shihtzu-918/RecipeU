# services/llm.py
"""
LLM 헬퍼 함수
"""
from typing import List, Dict, Optional


def create_system_prompt(
    user_profile: Optional[Dict],
    template: str,
    **kwargs
) -> str:
    """시스템 프롬프트 생성 (user_profile이 없으면 기본값 사용)"""
    # user_profile이 None이거나 빈 딕셔너리인 경우 기본값 사용
    if not user_profile:
        user_profile = {}

    user_name = user_profile.get("name", "사용자")
    allergies = user_profile.get("allergies", [])
    dislikes = user_profile.get("dislikes", user_profile.get("dislike", []))

    return template.format(
        user_name=user_name,
        allergies=", ".join(allergies) if allergies else "없음",
        dislike=", ".join(dislikes) if dislikes else "없음",
        **kwargs
    )


def format_chat_history(messages: List[Dict], max_items: int = 4) -> str:
    """채팅 히스토리 포맷팅"""
    history_text = "\n".join([
        f"{'사용자' if msg['role'] == 'user' else '어시스턴트'}: {msg['content'][:100]}"
        for msg in messages[-max_items:]
    ])
    return history_text