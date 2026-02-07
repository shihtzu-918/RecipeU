# features/voice/router.py
"""
음성 처리 SSE 라우터

엔드포인트:
  POST /stt           - 음성 → 텍스트 + 문장 완성도 분석 (JSON)
  POST /process-text  - 텍스트 → LLM → TTS (SSE 스트리밍)
  POST /process       - 음성 → STT → LLM → TTS (SSE, 기존 호환)
  POST /session       - 음성 세션 생성
  POST /save-history  - 음성 대화 기록 저장
  GET  /history/{id}  - 음성 대화 기록 조회
  GET  /health        - 상태 확인
"""
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import json
import logging

from features.voice.service import (
    process_voice_pipeline,
    process_text_pipeline,
    transcribe_and_analyze,
)
from models.mysql_db import create_session, add_chat_message, get_session_chats

logger = logging.getLogger("voice_router")

router = APIRouter()


@router.post("/stt")
async def stt_with_analysis(
    audio: UploadFile = File(..., description="VAD로 감지된 음성 파일"),
):
    """
    STT + 문장 완성도 분석

    Request:
        - audio: 음성 파일 (multipart/form-data)

    Response (JSON):
        {
            "text": "인식된 텍스트",
            "completeness": "COMPLETE" | "INCOMPLETE"
        }
    """
    audio_bytes = await audio.read()

    try:
        result = await transcribe_and_analyze(audio_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        logger.error(f"[STT] 에러 발생: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"text": "", "completeness": "INCOMPLETE", "error": str(e)},
            status_code=500,
        )


@router.post("/process-text")
async def process_text(
    text: str = Form(..., description="STT 완료된 최종 텍스트"),
    current_step: str = Form("", description="현재 조리 단계 설명"),
    current_cook: str = Form("", description="현재 요리 제목"),
    recipe_context: str = Form("", description="전체 레시피 정보"),
    step_index: int = Form(0, description="현재 단계 인덱스 (0부터)"),
    total_steps: int = Form(1, description="총 단계 수"),
    history: str = Form("[]", description="대화 기록 JSON ([{role, content}, ...])"),
):
    """
    텍스트 → LLM → TTS SSE 엔드포인트
    프론트에서 STT + 문장 완성도 처리 후 최종 텍스트를 보내면 사용

    Request:
        - text: 최종 사용자 텍스트
        - current_step: 현재 조리 단계 텍스트
        - current_cook: 현재 요리 제목
        - recipe_context: 전체 레시피 정보
        - step_index: 현재 단계 인덱스
        - total_steps: 총 단계 수
        - history: 대화 기록 JSON 문자열

    Response (SSE stream):
        - {"type": "llm", "intent": "...", "text": "...", "action": "..."}
        - {"type": "tts_chunk", "audio": "<base64>", "sample_rate": 32000}
        - {"type": "done"}
        - {"type": "error", "message": "..."}
    """
    # 대화 기록 파싱
    try:
        history_list = json.loads(history) if history else []
    except (json.JSONDecodeError, TypeError):
        history_list = []

    async def event_generator():
        async for event in process_text_pipeline(
            text,
            current_step,
            current_cook=current_cook,
            recipe_context=recipe_context,
            step_index=step_index,
            total_steps=total_steps,
            history=history_list
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/process")
async def process_voice(
    audio: UploadFile = File(..., description="VAD로 감지된 음성 파일"),
    current_step: str = Form("", description="현재 조리 단계 설명"),
    current_cook: str = Form("", description="현재 요리 제목"),
    recipe_context: str = Form("", description="전체 레시피 정보"),
    step_index: int = Form(0, description="현재 단계 인덱스 (0부터)"),
    total_steps: int = Form(1, description="총 단계 수"),
):
    """
    음성 처리 SSE 엔드포인트 (기존 호환용 - 전체 파이프라인)

    Request:
        - audio: 음성 파일 (multipart/form-data)
        - current_step: 현재 조리 단계 텍스트
        - current_cook: 현재 요리 제목
        - recipe_context: 전체 레시피 정보
        - step_index: 현재 단계 인덱스
        - total_steps: 총 단계 수

    Response (SSE stream):
        - {"type": "stt", "text": "..."}
        - {"type": "llm", "intent": "...", "text": "...", "action": "..."}
        - {"type": "tts_chunk", "audio": "<base64>", "sample_rate": 32000}
        - {"type": "done"}
        - {"type": "error", "message": "..."}
    """
    audio_bytes = await audio.read()

    async def event_generator():
        async for event in process_voice_pipeline(
            audio_bytes,
            current_step,
            current_cook=current_cook,
            recipe_context=recipe_context,
            step_index=step_index,
            total_steps=total_steps
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


## ====== 세션 & 히스토리 API ======

class SessionRequest(BaseModel):
    member_id: int = 2  # 비회원은 2로 고정

class ChatMessage(BaseModel):
    role: str       # 'user' or 'assistant'
    text: str

class SaveHistoryRequest(BaseModel):
    member_id: int = 2
    session_id: int
    messages: List[ChatMessage]


@router.post("/session")
async def create_voice_session(req: SessionRequest):
    """음성 세션 생성 → session_id 반환"""
    try:
        session = create_session(req.member_id)
        return {"session_id": session["session_id"]}
    except Exception as e:
        logger.error(f"[voice/session] 세션 생성 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/save-history")
async def save_voice_history(req: SaveHistoryRequest):
    """음성 대화 기록을 chatbot 테이블에 저장 (type=VOICE)"""
    try:
        saved = 0
        for msg in req.messages:
            add_chat_message(
                member_id=req.member_id,
                session_id=req.session_id,
                role=msg.role,
                text=msg.text,
                msg_type="VOICE",
            )
            saved += 1
        return {"saved": saved, "session_id": req.session_id}
    except Exception as e:
        logger.error(f"[voice/save-history] 저장 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/history/{session_id}")
async def get_voice_history(session_id: int):
    """음성 세션의 대화 기록 조회 (type=VOICE만)"""
    try:
        chats = get_session_chats(session_id)
        # VOICE 타입만 필터
        voice_chats = [c for c in chats if c.get("type") == "VOICE"]
        return {
            "session_id": session_id,
            "messages": [
                {"role": c["role"], "text": c["text"], "created_at": c.get("created_at", "")}
                for c in voice_chats
            ],
        }
    except Exception as e:
        logger.error(f"[voice/history] 조회 실패: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/health")
async def health_check():
    """Voice API 상태 확인"""
    return {"status": "ok", "service": "voice"}
