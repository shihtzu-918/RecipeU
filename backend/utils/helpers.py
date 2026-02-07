# utils/helpers.py
"""
기타 헬퍼 함수
"""
import uuid


def generate_session_id() -> str:
    """세션 ID 생성"""
    return str(uuid.uuid4())