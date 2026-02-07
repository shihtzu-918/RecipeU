# utils/parser.py
"""
파싱 유틸
"""
import re
from typing import Dict


def parse_recommendation(agent_text: str) -> Dict:
    """Agent 응답에서 요리 정보 추출"""
    title = None

    # 1. "오늘의 추천 요리는 XXX 입니다" 패턴
    title_match = re.search(
        r'오늘의\s+추천\s+요리[는은]\s+(.+?)\s+입니다',
        agent_text
    )
    if title_match:
        title = title_match.group(1).strip()

    # 2. 따옴표 또는 대괄호
    if not title:
        title_match = re.search(r'["\'\[\]](.+?)["\'\]\]]', agent_text)
        if title_match:
            title = title_match.group(1).strip()

    # 재료, 시간, 난이도 파싱
    ingredients_match = re.search(r'재료\s*[:：]\s*(.+?)(?=\n|소요시간|난이도|$)', agent_text)
    time_match = re.search(r'소요시간\s*[:：]\s*(.+?)(?=\n|난이도|$)', agent_text)
    level_match = re.search(r'난이도\s*[:：]\s*(.+?)(?=\n|$)', agent_text)

    return {
        "title": title or "알 수 없음",
        "ingredients": ingredients_match.group(1).strip() if ingredients_match else "",
        "cook_time": time_match.group(1).strip() if time_match else "30분",
        "level": level_match.group(1).strip() if level_match else "중급",
    }