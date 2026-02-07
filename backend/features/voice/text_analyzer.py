# features/voice/text_analyzer.py
"""
Kiwi 형태소 분석기를 사용한 문장 완성도 분석
- COMPLETE: 완성된 문장 (종결어미, 명사형 종결 등)
- INCOMPLETE: 미완성 문장 (연결어미, 조사 등)
"""
from kiwipiepy import Kiwi

_kiwi = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        print("[TextAnalyzer] Kiwi 형태소 분석기 로딩 중...")
        _kiwi = Kiwi()
        print("[TextAnalyzer] Kiwi 준비 완료")
    return _kiwi


def analyze_completeness(text: str) -> str:
    """
    문장의 완성도를 분석합니다.
    Return: "COMPLETE" (완성), "INCOMPLETE" (미완성)
    """
    clean_text = text.strip().rstrip(".,?!")

    if not clean_text:
        return "INCOMPLETE"

    try:
        kiwi = _get_kiwi()
        tokens = kiwi.tokenize(clean_text)
        if not tokens:
            return "INCOMPLETE"

        last_token = tokens[-1]
        tag = last_token.tag

        # --- [A. 완전한 문장 기준] ---
        # 1. 종결 어미 (EF): ~다, ~요, ~까, ~죠
        if tag.startswith("EF"):
            return "COMPLETE"

        # 2. 명사형 종결 (NNG, NNP, NNB, NR, NP): 단답형 명령
        if tag.startswith("N"):
            return "COMPLETE"

        # 3. 어근(XR) + 하(XSV) 생략된 경우 (예: "성공", "시작")
        if tag.startswith("XR"):
            return "COMPLETE"

        # --- [B. 불완전한 문장 기준] ---
        # 1. 연결 어미 (EC): ~고, ~면, ~서, ~는데
        if tag.startswith("EC"):
            return "INCOMPLETE"

        # 2. 조사 (J): ~은, ~는, ~이, ~가, ~을
        if tag.startswith("J"):
            return "INCOMPLETE"

        # 그 외(부사, 관형사 등)는 대화 중간일 확률이 높음
        return "INCOMPLETE"

    except Exception as e:
        print(f"[TextAnalyzer] 분석 에러: {e}")
        return "COMPLETE"  # 에러 시 안전하게 완성 처리
