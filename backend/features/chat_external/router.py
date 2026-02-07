# backend/features/chat_external/router.py
"""
외부 챗봇 WebSocket 라우터 - 단순 질문/답변
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import os
import logging
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/{session_id}")
async def external_chat_websocket(websocket: WebSocket, session_id: str):
    """외부 챗봇 WebSocket - 세션 저장 없이 단순 질문/답변"""
    await websocket.accept()
    logger.info(f"[External WS] Connected: {session_id}")
    
    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    
    logger.info(f"[External WS] API Key 확인: {bool(api_key)}")
    
    if not api_key:
        logger.error(f"[External WS] API 키 누락")
        await websocket.send_json({
            "type": "error",
            "message": "챗봇 설정이 완료되지 않았습니다."
        })
        await websocket.close()
        return
    
    try:
        try:
            from langchain_naver import ChatClovaX
            logger.info("[External WS] langchain_naver 사용")
        except ImportError:
            from langchain_community.chat_models import ChatClovaX
            logger.info("[External WS] langchain_community 사용")
        
        chat_model = ChatClovaX(
            model="HCX-003",
            temperature=0.7,
            max_tokens=500,
        )
        logger.info("[External WS] ChatClovaX 초기화 완료")
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "user_message":
                content = message.get("content", "").strip()
                
                if not content:
                    continue
                
                logger.info(f"[External WS] 사용자: {content}")
                
                # 생각 중 표시
                await websocket.send_json({
                    "type": "thinking",
                    "message": "생각 중..."
                })
                
                try:
                    # ✅ 단순하게 현재 메시지만 LLM에 전달
                    response = chat_model.invoke(content)
                    assistant_message = response.content.strip()
                    
                    logger.info(f"[External WS] 응답: {assistant_message[:100]}...")
                    
                    # 응답 전송
                    await websocket.send_json({
                        "type": "assistant_message",
                        "content": assistant_message
                    })
                
                except Exception as e:
                    logger.error(f"[External WS] LLM 오류: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": "응답 생성 중 오류가 발생했습니다."
                    })
    
    except WebSocketDisconnect:
        logger.info(f"[External WS] Disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"[External WS] 에러: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "오류가 발생했습니다."
            })
        except:
            pass
    
    finally:
        logger.info(f"[External WS] Closed: {session_id}")


@router.get("/health")
async def health_check():
    """헬스 체크"""
    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    return {
        "status": "healthy",
        "clova_configured": bool(api_key)
    }