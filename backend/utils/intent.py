# backend/utils/intent.py
"""
의도 감지 유틸
"""
from typing import List  # 이 줄 추가!


class Intent:
    NEXT = "next_step"
    PREV = "prev_step"
    SUB_ING = "substitute_ingredient"
    SUB_TOOL = "substitute_tool"
    FAILURE = "failure"
    UNKNOWN = "unknown"


def detect_intent(text: str) -> str:
    """사용자 의도 감지"""
    t = (text or "").strip().lower().replace(" ", "")

    # 다음/이전
    if any(k in t for k in ["다음", "넘겨", "다음으로", "다음단계"]):
        return Intent.NEXT
    if any(k in t for k in ["이전", "뒤로", "전단계", "이전단계"]):
        return Intent.PREV

    # 실패
    if any(k in t for k in ["탔", "타버", "눌었", "쏟", "엎", "망했", "실패"]):
        return Intent.FAILURE

    # 대체재료/도구
    has_no = any(k in t for k in ["없어", "없는데", "없다"])
    if has_no and any(k in t for k in ["재료", "대체", "대신"]):
        return Intent.SUB_ING
    if has_no and any(k in t for k in ["도구", "기구", "냄비", "오븐"]):
        return Intent.SUB_TOOL

    return Intent.UNKNOWN


def extract_constraints(text: str) -> List[str]:
    """제약 조건 추출"""
    constraints = []
    content = text.replace(" ", "").lower()
    
    if any(k in content for k in ["초보", "쉬운", "간단"]):
        constraints.append("쉬운")
    if any(k in content for k in ["빠른", "빨리"]):
        constraints.append("빠른")
    if any(k in content for k in ["건강", "다이어트"]):
        constraints.append("저칼로리")
    
    return constraints