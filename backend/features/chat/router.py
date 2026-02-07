# backend/features/chat/router.py
"""
Chat Agent WebSocket 라우터 - Adaptive RAG + 레시피 수정
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict
import json
import asyncio
import time
from langchain_naver import ChatClovaX

from core.websocket import manager
from core.dependencies import get_rag_system
from features.chat.agent import create_chat_agent, _node_timings
from models.mysql_db import create_session, add_chat_message

logger = logging.getLogger(__name__)

router = APIRouter()

chat_sessions: Dict[str, dict] = {}


def _print_timing_summary(total_ms: float):
    if not _node_timings:
        return
    logger.info("┌─────────────────────────────────────────┐")
    logger.info("│          Node Timing Summary            │")
    logger.info("├─────────────────────────────────────────┤")
    for name, ms in _node_timings.items():
        bar_len = int(ms / max(max(_node_timings.values()), 1) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        pct = (ms / total_ms * 100) if total_ms > 0 else 0
        sec = ms / 1000
        logger.info(f"│  {name:<18} {bar} {sec:>5.1f}초 ({pct:>4.1f}%) │")
    logger.info("├─────────────────────────────────────────┤")
    total_sec = total_ms / 1000
    logger.info(f"│  {'TOTAL':<18} {'':20} {total_sec:>5.1f}초        │")
    logger.info("└─────────────────────────────────────────┘")
    _node_timings.clear()


async def handle_recipe_modification(websocket: WebSocket, session: Dict, user_input: str):
    """레시피 수정 처리 (기존 레시피를 사용자 요청대로 수정)"""
    logger.info("[WS] 🔧 레시피 수정 모드 시작")
    
    # 히스토리에서 원본 레시피와 이미지 찾기
    original_recipe_content = None
    original_image = None
    
    for msg in session["messages"]:
        if msg["role"] == "assistant" and "[" in msg["content"]:
            original_recipe_content = msg["content"]
            original_image = msg.get("image", "")  
            logger.info(f"[WS] 원본 레시피 발견")
            logger.info(f"[WS] 원본 이미지: {original_image[:60] if original_image else '없음'}...")
            break
    
    if not original_recipe_content:
        logger.warning("[WS] 원본 레시피 없음 → 일반 대화로 처리")
        return False
    
    await websocket.send_json({"type": "thinking"})
    
    modification_prompt = f"""당신은 레시피 수정 전문가입니다.

원본 레시피:
{original_recipe_content}

사용자 요청: {user_input}

위 레시피의 제목을 유지하면서, 사용자 요청을 반영해서 수정해주세요.
**중요: 새로운 레시피를 만들지 말고, 위 레시피만 수정하세요!**

수정 규칙:
- 제목은 반드시 유지하세요
- "더 맵게" → 고추 계열 재료 양 2배 증가
- "덜 달게" → 설탕 양 50% 감소
- "덜 짜게" → 간장/소금 50% 감소

같은 형식으로 응답:
[제목]
⏱️ 시간 | 📊 난이도 | 👥 인분

재료
- ...

조리법
1. ..."""
    
    llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=1500)
    
    try:
        result = llm.invoke(modification_prompt)
        modified_recipe = result.content.strip()
        
        logger.info("[WS] 레시피 수정 완료")
        
        # 히스토리에 추가 (이미지 포함!)
        session["messages"].append({
            "role": "assistant",
            "content": modified_recipe,
            "image": original_image  # 원본 이미지 유지
        })
        
        # WebSocket 응답 (이미지 포함 + hideImage)
        await websocket.send_json({
            "type": "agent_message",
            "content": modified_recipe,
            "image": original_image,  # 데이터 전달
            "hideImage": True  # UI에는 안 보이게
        })
        
        return True
        
    except Exception as e:
        logger.error(f"[WS] ❌ 레시피 수정 실패: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": "레시피 수정 중 오류가 발생했습니다."
        })
        return True


@router.websocket("/ws/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: str,
    rag_system = Depends(get_rag_system),
):
    await websocket.accept()
    logger.info(f"[WS] Connected: {session_id}")

    if not rag_system:
        logger.warning("[WS] RAG 시스템 없음")
        await websocket.send_json({"type": "error", "message": "RAG 시스템을 사용할 수 없습니다."})
        await websocket.close()
        return

    try:
        agent = create_chat_agent(rag_system)
        if not agent:
            raise ValueError("Agent 생성 실패")
        logger.info("[WS] Adaptive RAG Agent 생성 완료")
    except Exception as e:
        logger.error(f"[WS] Agent 생성 에러: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Agent 생성 실패: {str(e)}"})
        await websocket.close()
        return

    manager.active_connections[session_id] = websocket

    # DB 세션은 init_context에서 member_id를 받은 후 생성
    db_session_id = None
    member_id = 0  # 기본값, init_context에서 업데이트

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "messages": [],
            "user_constraints": {},
            "last_documents": [],
            "last_agent_response": "",
            "db_session_id": None,
            "member_id": 0,
        }

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            logger.info(f"[WS] 메시지 수신: {msg_type}")

            if msg_type == "init_context":
                member_info = message.get("member_info", {})
                initial_history = message.get("initial_history", [])

                chat_sessions[session_id]["user_constraints"] = member_info

                # ✅ member_id 추출 및 DB 세션 생성
                mid = member_info.get("member_id")
                logger.info(f"[WS] init_context 수신: member_id={mid} (type: {type(mid).__name__})")

                # member_id를 int로 변환 (숫자 또는 숫자 문자열 모두 처리)
                try:
                    member_id = int(mid) if mid is not None else 0
                except (ValueError, TypeError):
                    member_id = 0

                if member_id > 0:
                    chat_sessions[session_id]["member_id"] = member_id

                    # DB 세션이 아직 없으면 생성
                    if not chat_sessions[session_id].get("db_session_id"):
                        try:
                            from models.mysql_db import create_session
                            db_result = create_session(member_id=member_id)
                            db_session_id = db_result.get("session_id") if db_result else None
                            chat_sessions[session_id]["db_session_id"] = db_session_id

                            # 클라이언트로 db_session_id 전송
                            if db_session_id:
                                await websocket.send_json({
                                    "type": "session_initialized",
                                    "session_id": session_id,
                                    "db_session_id": db_session_id
                                })
                                logger.info(f"[WS] DB 세션 생성 완료: db_session_id={db_session_id}, member_id={member_id}")
                            else:
                                logger.warning(f"[WS] DB 세션 생성 결과가 None: db_result={db_result}")
                        except Exception as e:
                            logger.error(f"[WS] DB 세션 생성 실패: {e}", exc_info=True)
                else:
                    logger.warning(f"[WS] member_id가 0 또는 유효하지 않음: {mid}")

                # 초기 히스토리 설정 (레시피 수정 모드용)
                if initial_history:
                    chat_sessions[session_id]["messages"].extend(initial_history)
                    logger.info(f"[WS] 초기 히스토리 {len(initial_history)}개 추가")

                logger.info(f"[WS] 컨텍스트 설정: {member_info.get('names', [])}, member_id={member_id}")
                continue

            elif msg_type == "user_message":
                content = message.get("content", "")
                is_modification = message.get("is_recipe_modification", False)  
                
                logger.info(f"[WS] 사용자 메시지: {content}")
                logger.info(f"[WS] 레시피 수정 모드: {is_modification}")

                start_time = time.time()
                
                # 사용자 메시지 히스토리에 추가
                chat_sessions[session_id]["messages"].append({
                    "role": "user",
                    "content": content
                })
                
                # 레시피 수정 모드 처리
                if is_modification:
                    modification_success = await handle_recipe_modification(
                        websocket, 
                        chat_sessions[session_id], 
                        content
                    )
                    
                    if modification_success:
                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 레시피 수정 완료 (총 {total_sec:.1f}초)")
                        continue  # 일반 대화 로직 건너뜀
                    
                    # False면 일반 대화로 계속 진행
                    logger.info("[WS] 일반 대화로 전환")
                
                # ✅ 일반 대화 모드
                chat_history = [
                    f"{msg['role']}: {msg['content']}" 
                    for msg in chat_sessions[session_id]["messages"]
                ]

                await websocket.send_json({"type": "thinking", "message": "생각 중..."})

                agent_state = {
                    "question": content,
                    "original_question": content,
                    "chat_history": chat_history,
                    "documents": [],
                    "generation": "",
                    "web_search_needed": "no",
                    "user_constraints": chat_sessions[session_id]["user_constraints"],
                    "constraint_warning": ""
                }

                async def progress_notifier():
                    steps = [
                        (0, "쿼리 재작성 중..."), 
                        (3, "레시피 검색 중..."), 
                        (6, "관련성 평가 중..."), 
                        (10, "답변 생성 중..."), 
                        (15, "거의 완료...")
                    ]
                    for delay, msg in steps:
                        await asyncio.sleep(delay if delay == 0 else 3)
                        if time.time() - start_time < 20:
                            await websocket.send_json({
                                "type": "progress", 
                                "message": f"{msg} ({int(time.time() - start_time)}초)"
                            })
                        else:
                            break

                notifier_task = asyncio.create_task(progress_notifier())

                try:
                    _node_timings.clear()

                    async def run_agent():
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, agent.invoke, agent_state)

                    result = await asyncio.wait_for(run_agent(), timeout=20.0)

                    total_ms = (time.time() - start_time) * 1000
                    _print_timing_summary(total_ms)

                    # 캐시 저장
                    agent_docs = result.get("documents", [])
                    agent_response = result.get("generation", "")

                    if agent_docs:
                        chat_sessions[session_id]["last_documents"] = [
                            {
                                "content": doc.page_content,
                                "title": doc.metadata.get("title", ""),
                                "cook_time": doc.metadata.get("cook_time", ""),
                                "level": doc.metadata.get("level", ""),
                                "recipe_id": doc.metadata.get("recipe_id", ""),
                            }
                            for doc in agent_docs
                        ]
                        logger.info(f"[WS] 세션 캐시 저장: {len(agent_docs)}개 문서")

                    if agent_response and agent_response != "NOT_RECIPE_RELATED":
                        chat_sessions[session_id]["last_agent_response"] = agent_response
                        logger.info(f"[WS] Agent 답변 캐시: {agent_response[:60]}...")

                    response = agent_response or "답변을 생성할 수 없습니다."

                    if response == "NOT_RECIPE_RELATED":
                        logger.info("[WS] 요리 무관 대화 감지")
                        not_recipe_msg = "죄송합니다. 저는 요리 레시피만 도와드릴 수 있어요! 🍳\n일반적인 질문은 다른 AI 챗봇을 이용해주세요."
                        
                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": not_recipe_msg
                        })
                        
                        await websocket.send_json({
                            "type": "not_recipe_related",
                            "content": not_recipe_msg
                        })
                        
                        total_sec = total_ms / 1000
                        logger.info(f"[WS] ✅ 응답 완료 (총 {total_sec:.1f}초)")
                        continue

                    chat_sessions[session_id]["messages"].append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    await websocket.send_json({
                        "type": "agent_message", 
                        "content": response
                    })
                    
                    total_sec = total_ms / 1000
                    logger.info(f"[WS] ✅ 응답 완료 (총 {total_sec:.1f}초)")

                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"[WS] ⏱️ Agent 타임아웃 ({elapsed:.1f}초)")
                    _print_timing_summary(elapsed * 1000)
                    
                    await websocket.send_json({
                        "type": "agent_message",
                        "content": f"죄송합니다. 응답 시간이 너무 오래 걸렸어요 ({int(elapsed)}초). 다시 시도해주세요."
                    })
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(f"[WS] ⚠️ Agent 실행 에러 ({elapsed:.1f}초): {e}", exc_info=True)
                    _print_timing_summary(elapsed * 1000)
                    
                    await websocket.send_json({
                        "type": "error", 
                        "message": f"오류가 발생했습니다 ({int(elapsed)}초). 다시 시도해주세요."
                    })
                    
                finally:
                    notifier_task.cancel()
                    try:
                        await notifier_task
                    except asyncio.CancelledError:
                        pass

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[WS] 에러: {e}", exc_info=True)
    finally:
        manager.disconnect(session_id)
        logger.info(f"[WS] Closed: {session_id}")


@router.get("/session/{session_id}")
async def get_chat_session(session_id: str):
    logger.info(f"[Chat API] 세션 조회: {session_id}")
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    session = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "messages": session.get("messages", []),
        "user_constraints": session.get("user_constraints", {})
    }