# features/cooking/router.py
"""
Cooking Agent WebSocket 라우터
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Dict
import os
import tempfile

from core.websocket import manager
from core.dependencies import get_rag_system
from core.exceptions import SessionNotFoundError
from features.cooking.agent import CookingAgent
from features.cooking.session import CookingSession
from models.mysql_db import (
    create_session, add_chat_message, save_voice
)


router = APIRouter()

# 세션별 Agent 저장 (agent + metadata)
cooking_sessions: Dict[str, dict] = {}


@router.websocket("/ws/{session_id}")
async def cooking_websocket(
    websocket: WebSocket,
    session_id: str,
    rag_system = Depends(get_rag_system)
):
    """조리모드 Agent WebSocket - voice 테이블 저장 포함"""

    if not rag_system:
        await websocket.close(code=1011, reason="RAG system not available")
        return

    await manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "init":
                # 레시피 설정
                recipe = data.get("recipe")
                member_id = data.get("member_id", 0)
                if member_id and str(member_id).isdigit():
                    member_id = int(member_id)
                else:
                    member_id = 0

                if not recipe:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": "레시피가 필요합니다."
                    })
                    continue

                # CookingAgent 생성
                try:
                    cooking_session = CookingSession(rag=rag_system)
                    agent = CookingAgent(rag_system, cooking_session)
                    agent.set_recipe(recipe)

                    # MySQL 세션 생성 (로그인 사용자만)
                    db_session_id = None
                    if member_id > 0:
                        try:
                            db_session = create_session(member_id)
                            db_session_id = db_session.get("session_id")
                            print(f"[Cook WS] MySQL 세션 생성: {db_session_id}")
                        except Exception as e:
                            print(f"[Cook WS] MySQL 세션 생성 실패: {e}")

                    # 세션 정보 저장 (agent + metadata)
                    cooking_sessions[session_id] = {
                        "agent": agent,
                        "session": cooking_session,
                        "member_id": member_id,
                        "db_session_id": db_session_id
                    }

                    # 첫 단계 안내
                    steps = recipe.get("steps", [])
                    if steps:
                        first_step = steps[0]
                        msg = f"{first_step.get('no', 1)}단계: {first_step.get('desc','')}"
                        audio_path = cooking_session.generate_tts(msg)

                        # TTS를 voice 테이블에 저장
                        if db_session_id and member_id > 0:
                            try:
                                # 먼저 chatbot 메시지 추가 (AGENT 역할)
                                chat_result = add_chat_message(
                                    member_id=member_id,
                                    session_id=db_session_id,
                                    role="AGENT",
                                    text=msg,
                                    msg_type="DEFAULT"
                                )
                                chat_id = chat_result.get("chat_id")
                                # voice 저장
                                save_voice(
                                    chat_id=chat_id,
                                    member_id=member_id,
                                    voice_type="TTS",
                                    context=msg,
                                    voice_file=audio_path
                                )
                            except Exception as e:
                                print(f"[Cook WS] voice 저장 실패: {e}")

                        await manager.send_message(session_id, {
                            "type": "cook_response",
                            "text": msg,
                            "step_index": 0,
                            "total_steps": len(steps),
                            "audio_url": f"/api/cook/audio/{os.path.basename(audio_path)}"
                        })

                except Exception as e:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": f"조리 Agent 초기화 실패: {str(e)}"
                    })

            elif data["type"] == "text_input":
                # 텍스트 입력
                session_data = cooking_sessions.get(session_id)
                if not session_data:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": "조리 세션이 없습니다."
                    })
                    continue

                agent = session_data["agent"]
                member_id = session_data.get("member_id", 0)
                db_session_id = session_data.get("db_session_id")
                user_text = data.get("text", "")

                # 사용자 메시지 저장
                user_chat_id = None
                if db_session_id and member_id > 0:
                    try:
                        chat_result = add_chat_message(
                            member_id=member_id,
                            session_id=db_session_id,
                            role="USER",
                            text=user_text,
                            msg_type="DEFAULT"
                        )
                        user_chat_id = chat_result.get("chat_id")
                    except Exception as e:
                        print(f"[Cook WS] 사용자 메시지 저장 실패: {e}")

                # Agent 처리
                await manager.send_message(session_id, {
                    "type": "thinking",
                    "message": "처리 중..."
                })

                result = await agent.handle_input(user_text)

                # Agent 응답 저장 + TTS voice 저장
                if db_session_id and member_id > 0:
                    try:
                        chat_result = add_chat_message(
                            member_id=member_id,
                            session_id=db_session_id,
                            role="AGENT",
                            text=result["response"],
                            msg_type="DEFAULT"
                        )
                        chat_id = chat_result.get("chat_id")
                        save_voice(
                            chat_id=chat_id,
                            member_id=member_id,
                            voice_type="TTS",
                            context=result["response"],
                            voice_file=result["audio_path"]
                        )
                    except Exception as e:
                        print(f"[Cook WS] Agent 응답 저장 실패: {e}")

                await manager.send_message(session_id, {
                    "type": "cook_response",
                    "text": result["response"],
                    "step_index": result["current_step"],
                    "total_steps": result["total_steps"],
                    "audio_url": f"/api/cook/audio/{os.path.basename(result['audio_path'])}"
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        if session_id in cooking_sessions:
            del cooking_sessions[session_id]
    except Exception as e:
        print(f"Cooking WebSocket 오류: {e}")
        import traceback
        traceback.print_exc()
        manager.disconnect(session_id)


@router.post("/audio/{session_id}")
async def upload_audio(
    session_id: str,
    file: UploadFile = File(...)
):
    """음성 파일 업로드 및 처리 - STT/TTS voice 테이블 저장"""
    session_data = cooking_sessions.get(session_id)

    if not session_data:
        raise SessionNotFoundError(session_id)

    agent = session_data["agent"]
    member_id = session_data.get("member_id", 0)
    db_session_id = session_data.get("db_session_id")

    # 임시 파일로 저장
    suffix = "." + file.filename.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Agent 처리 (STT → 응답 생성 → TTS)
        result = await agent.handle_audio(tmp_path)

        # STT 결과 저장 (사용자 음성 입력)
        if db_session_id and member_id > 0 and result.get("user_text"):
            try:
                # 사용자 메시지 저장
                user_chat = add_chat_message(
                    member_id=member_id,
                    session_id=db_session_id,
                    role="USER",
                    text=result["user_text"],
                    msg_type="DEFAULT"
                )
                user_chat_id = user_chat.get("chat_id")
                # STT voice 저장
                save_voice(
                    chat_id=user_chat_id,
                    member_id=member_id,
                    voice_type="STT",
                    context=result["user_text"],
                    voice_file=tmp_path
                )

                # Agent 응답 저장
                agent_chat = add_chat_message(
                    member_id=member_id,
                    session_id=db_session_id,
                    role="AGENT",
                    text=result["response"],
                    msg_type="DEFAULT"
                )
                agent_chat_id = agent_chat.get("chat_id")
                # TTS voice 저장
                save_voice(
                    chat_id=agent_chat_id,
                    member_id=member_id,
                    voice_type="TTS",
                    context=result["response"],
                    voice_file=result["audio_path"]
                )
            except Exception as e:
                print(f"[Cook Audio] voice 저장 실패: {e}")

        return {
            "text": result["response"],
            "user_text": result.get("user_text", ""),
            "step_index": result["current_step"],
            "total_steps": result["total_steps"],
            "audio_url": f"/api/cook/audio/{os.path.basename(result['audio_path'])}"
        }
    finally:
        os.unlink(tmp_path)


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """TTS 오디오 파일 제공"""
    audio_dir = os.path.join(tempfile.gettempdir(), "cook_tts")
    file_path = os.path.join(audio_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/wav")