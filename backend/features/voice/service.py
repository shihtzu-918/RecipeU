# features/voice/service.py
"""
음성 처리 서비스 (STT → LLM → TTS)

STT: Naver Clova Speech API (기존 자체 STT 서버 대체)
문장 완성도: Kiwi 형태소 분석기
LLM/TTS: 기존 외부 API 유지

Intent 종류 (LLM 프롬프트 기준):
  Next, Prev, Finish,
  Missing Ingredient, Missing Tool, Failure,
  Out of Scope
"""
import httpx
import base64
from enum import Enum
from typing import AsyncGenerator, Dict, Any, List

from app.config import settings
from features.voice.clova_speech_client import ClovaSpeechClient
from features.voice.text_analyzer import analyze_completeness


# ============================================================================
# 외부 API 엔드포인트 (백엔드에서만 관리)
# ============================================================================
LLM_BASE_URL = "http://213.173.107.104:12172"
TTS_BASE_URL = "http://213.173.107.104:12171"

LLM_ENDPOINT = "/classify"
TTS_ENDPOINT = "/synthesize/stream"

# API Key 헤더 (RunPod 서버 인증용)
API_KEY_HEADER = "X-API-Key"


# ============================================================================
# Clova Speech STT 클라이언트 (싱글턴)
# ============================================================================
_stt_client = None


def _get_stt_client() -> ClovaSpeechClient:
    global _stt_client
    if _stt_client is None:
        invoke_url = settings.CLOVA_STT_INVOKE_URL
        secret_key = settings.CLOVA_STT_SECRET_KEY
        if not invoke_url or not secret_key:
            raise RuntimeError("CLOVA_STT_INVOKE_URL / CLOVA_STT_SECRET_KEY가 .env에 설정되지 않았습니다.")
        _stt_client = ClovaSpeechClient(invoke_url, secret_key)
        print(f"[STT] Clova Speech 클라이언트 초기화 완료 (URL: {invoke_url[:50]}...)")
    return _stt_client


# ============================================================================
# Intent 매핑 (LLM 출력 → 내부 코드)
# ============================================================================
class Intent(str, Enum):
    NEXT = "next_step"
    PREV = "prev_step"
    FINISH = "finish"
    SUB_ING = "substitute_ingredient"
    SUB_TOOL = "substitute_tool"
    FAILURE = "failure"
    OUT_OF_SCOPE = "out_of_scope"


# LLM이 반환하는 Intent 문자열 → 내부 Intent enum
# raw_output 형태(Missing Ingredient)와 파싱된 형태(substitute_ingredient) 모두 지원
INTENT_MAP = {
    # raw_output 형태 (LLM 원본)
    "Next": Intent.NEXT,
    "Prev": Intent.PREV,
    "Finish": Intent.FINISH,
    "Missing Ingredient": Intent.SUB_ING,
    "Missing Tool": Intent.SUB_TOOL,
    "Failure": Intent.FAILURE,
    "Out of Scope": Intent.OUT_OF_SCOPE,
    # API 파싱된 형태 (소문자/언더스코어)
    "next_step": Intent.NEXT,
    "prev_step": Intent.PREV,
    "finish": Intent.FINISH,
    "substitute_ingredient": Intent.SUB_ING,
    "substitute_tool": Intent.SUB_TOOL,
    "failure": Intent.FAILURE,
    "out_of_scope": Intent.OUT_OF_SCOPE,
}


def map_intent(raw_intent: str) -> Intent:
    """LLM 원본 intent 문자열을 내부 Intent로 변환"""
    return INTENT_MAP.get(raw_intent, Intent.OUT_OF_SCOPE)


# ============================================================================
# STT (Clova Speech API) + 문장 완성도 분석
# ============================================================================
async def transcribe_audio(audio_bytes: bytes) -> str:
    """STT: 음성 바이트 → 텍스트 (Clova Speech API)"""
    client = _get_stt_client()
    return await client.transcribe(audio_bytes)


async def transcribe_and_analyze(audio_bytes: bytes) -> Dict[str, Any]:
    """
    STT + Kiwi 문장 완성도 분석

    Returns:
        {
            "text": "인식된 텍스트",
            "completeness": "COMPLETE" | "INCOMPLETE"
        }
    """
    text = await transcribe_audio(audio_bytes)

    if not text:
        return {"text": "", "completeness": "INCOMPLETE"}

    completeness = analyze_completeness(text)
    print(f"[STT+분석] 인식: \"{text}\" → [{completeness}]")

    return {"text": text, "completeness": completeness}


# ============================================================================
# LLM
# ============================================================================
async def classify_intent(
    user_text: str,
    current_step: str,
    current_cook: str = "",
    recipe_context: str = "",
    history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    LLM: 텍스트 분류 및 응답 생성

    LLM 응답 형식:
      {"Intent": "Next", "Response": null}
      {"Intent": "Missing Ingredient", "Response": "대체재료는 ..."}
    """
    payload = {
        "text": user_text,
        "current_step": current_step,
        "current_cook": current_cook,
        "recipe_context": recipe_context,
    }
    history_preview = f"{len(history)} turns" if history else "none"
    print(
        f"[LLM 요청] text='{user_text[:80]}...', current_step='{current_step[:80]}...', "
        f"current_cook='{current_cook}', recipe_context='{recipe_context}', "
        f"history={history_preview}"
    )
    if history:
        payload["history"] = history

    headers = {API_KEY_HEADER: settings.RECIPEU_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LLM_BASE_URL}{LLM_ENDPOINT}",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()

        print(f"[LLM 원본 응답] keys={list(result.keys())}, 전체={result}")

        # LLM 응답 파싱 (대소문자 키 모두 지원)
        raw_intent = (
            result.get("Intent")
            or result.get("intent")
            or result.get("label")
            or result.get("category")
            or ""
        )
        response_text = (
            result.get("Response")
            or result.get("responseText")
            or result.get("response")
            or result.get("text")
            or result.get("answer")
            or ""
        )

        print(f"[LLM 파싱] raw_intent='{raw_intent}', response_text='{str(response_text)[:80]}...'")

        intent = map_intent(raw_intent)
        print(f"[LLM 매핑] '{raw_intent}' → {intent.value}")

        return {
            "intent": intent,
            "response_text": response_text.strip() if response_text else ""
        }


# ============================================================================
# TTS (스트리밍)
# ============================================================================
async def synthesize_speech_stream(
    text: str,
    tone: str = "kiwi",
    text_lang: str = "ko",
    speed_factor: float = 1.0
) -> AsyncGenerator[Dict[str, Any], None]:
    """TTS: 텍스트 → 음성 스트림 (청크 단위로 yield)"""
    headers = {API_KEY_HEADER: settings.RECIPEU_API_KEY}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{TTS_BASE_URL}{TTS_ENDPOINT}",
            json={
                "text": text,
                "tone": tone,
                "text_lang": text_lang,
                "speed_factor": speed_factor
            },
            headers=headers
        ) as response:
            response.raise_for_status()

            sample_rate = int(response.headers.get("X-Sample-Rate", "32000"))
            first_chunk = True

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    audio_base64 = base64.b64encode(chunk).decode("utf-8")

                    if first_chunk:
                        yield {"audio": audio_base64, "sample_rate": sample_rate}
                        first_chunk = False
                    else:
                        yield {"audio": audio_base64}


# ============================================================================
# Intent별 분기 처리 (LLM → TTS) - 공통 로직
# ============================================================================
async def _process_intent(
    intent: Intent,
    llm_text: str,
    step_index: int,
    total_steps: int
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Intent별 분기 처리 + TTS 스트리밍

    Intent별 처리:
      NEXT: 안내 + TTS + step 변경. 마지막 step이면 종료 안내 + TTS
      PREV: 안내 + TTS + step 변경. 첫 step이면 안내 + TTS
      FINISH: voice 모드 종료, 안내 + TTS
      SUB_ING / SUB_TOOL / FAILURE: LLM 응답 + TTS
      OUT_OF_SCOPE: LLM 응답 + TTS
    """
    print(f"[파이프라인] intent={intent}, intent.value={intent.value}, llm_text='{llm_text[:80]}...' step_index={step_index}, total_steps={total_steps}")
    is_first_step = (step_index <= 0)
    is_last_step = (step_index >= total_steps - 1)

    if intent == Intent.NEXT:
        if is_last_step:
            response_text = "마지막 단계예요. 음성 모드를 종료합니다."
            yield {
                "type": "llm",
                "intent": intent.value,
                "text": response_text,
                "action": "end_cooking",
                "delay_seconds": 3
            }
            try:
                async for tts_chunk in synthesize_speech_stream(response_text):
                    yield {"type": "tts_chunk", **tts_chunk}
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
                yield {"type": "error", "message": "음성 합성 중 오류가 발생했습니다."}
        else:
            response_text = "다음 단계로 넘어갈게요."
            yield {
                "type": "llm",
                "intent": intent.value,
                "text": response_text,
                "action": "step_change"
            }
            try:
                async for tts_chunk in synthesize_speech_stream(response_text):
                    yield {"type": "tts_chunk", **tts_chunk}
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
                yield {"type": "error", "message": "음성 합성 중 오류가 발생했습니다."}

        yield {"type": "done"}
        return

    elif intent == Intent.PREV:
        if is_first_step:
            response_text = "이전 단계로 이동할 수 없습니다."
            yield {
                "type": "llm",
                "intent": intent.value,
                "text": response_text,
                "action": "blocked"
            }
            try:
                async for tts_chunk in synthesize_speech_stream(response_text):
                    yield {"type": "tts_chunk", **tts_chunk}
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
                yield {"type": "error", "message": "음성 합성 중 오류가 발생했습니다."}
        else:
            response_text = "이전 단계로 돌아갈게요."
            yield {
                "type": "llm",
                "intent": intent.value,
                "text": response_text,
                "action": "step_change"
            }
            try:
                async for tts_chunk in synthesize_speech_stream(response_text):
                    yield {"type": "tts_chunk", **tts_chunk}
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
                yield {"type": "error", "message": "음성 합성 중 오류가 발생했습니다."}

        yield {"type": "done"}
        return

    elif intent == Intent.FINISH:
        response_text = "음성모드를 종료합니다."
        yield {
            "type": "llm",
            "intent": intent.value,
            "text": response_text,
            "action": "finish",
            "delay_seconds": 3
        }
        try:
            async for tts_chunk in synthesize_speech_stream(response_text):
                yield {"type": "tts_chunk", **tts_chunk}
        except Exception as e:
            print(f"TTS 오류: {str(e)}")
            yield {"type": "error", "message": "음성 합성 중 오류가 발생했습니다."}
        yield {"type": "done"}
        return

    elif intent in (Intent.SUB_ING, Intent.SUB_TOOL, Intent.FAILURE, Intent.OUT_OF_SCOPE):
        response_text = llm_text if llm_text else "처리할 수 없어요."
        yield {
            "type": "llm",
            "intent": intent.value,
            "text": response_text
        }

    else:
        response_text = "답변이 불가능한 질문이에요."
        yield {
            "type": "llm",
            "intent": intent.value,
            "text": response_text
        }

    # TTS (SUB_ING, SUB_TOOL, FAILURE, OUT_OF_SCOPE만 여기 도달)
    try:
        async for tts_chunk in synthesize_speech_stream(response_text):
            yield {"type": "tts_chunk", **tts_chunk}

        yield {"type": "done"}

    except Exception as e:
        print(f"TTS 오류: {str(e)}")
        yield {"type": "error", "message": "음성 합성 중 오류가 발생했습니다."}


# ============================================================================
# 텍스트 기반 파이프라인: (이미 STT 완료된) 텍스트 → LLM → TTS
# ============================================================================
async def process_text_pipeline(
    user_text: str,
    current_step: str,
    current_cook: str = "",
    recipe_context: str = "",
    step_index: int = 0,
    total_steps: int = 1,
    history: List[Dict[str, str]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    텍스트 기반 파이프라인: 텍스트 → LLM → TTS (SSE 스트리밍)
    프론트에서 STT + 문장 완성도를 처리한 후 최종 텍스트를 보내면 사용
    """
    # ── 1. LLM ──
    try:
        llm_result = await classify_intent(
            user_text,
            current_step,
            current_cook=current_cook,
            recipe_context=recipe_context,
            history=history
        )
        intent = llm_result["intent"]
        llm_text = llm_result["response_text"]

    except Exception as e:
        print(f"LLM 오류: {str(e)}")
        yield {"type": "error", "message": "답변 생성 중 오류가 발생했습니다."}
        return

    # ── 2. Intent별 분기 처리 + TTS ──
    async for event in _process_intent(intent, llm_text, step_index, total_steps):
        yield event


# ============================================================================
# (expired)전체 파이프라인: STT → LLM → (조건부) TTS (기존 호환용)
# ============================================================================
async def process_voice_pipeline(
    audio_bytes: bytes,
    current_step: str,
    current_cook: str = "",
    recipe_context: str = "",
    step_index: int = 0,
    total_steps: int = 1
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    전체 파이프라인: STT → LLM → TTS (SSE 스트리밍)
    기존 /process 엔드포인트 호환용
    """

    # ── 1. STT ──
    try:
        user_text = await transcribe_audio(audio_bytes)

        if not user_text:
            yield {"type": "error", "message": "음성을 인식하지 못했어요."}
            return

        yield {"type": "stt", "text": user_text}

    except Exception as e:
        print(f"STT 오류: {str(e)}")
        yield {"type": "error", "message": "음성 인식 중 오류가 발생했습니다."}
        return

    # ── 2. LLM → TTS ──
    async for event in process_text_pipeline(
        user_text,
        current_step,
        current_cook=current_cook,
        recipe_context=recipe_context,
        step_index=step_index,
        total_steps=total_steps
    ):
        yield event