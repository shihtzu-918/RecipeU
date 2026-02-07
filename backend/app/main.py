# backend/app/main.py
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from core.dependencies import get_rag_system
from features.chat.router import router as chat_router
from features.chat_external.router import router as chat_external_router
from features.recipe.router import router as recipe_router
from features.cooking.router import router as cooking_router
from features.user.router import router as user_router
from features.auth.router import router as auth_router
from features.mypage.router import router as mypage_router, init_utensils
from features.weather.router import router as weather_router
from features.ranking.router import router as ranking_router, load_today_ranking_cache
from features.voice.router import router as voice_router
from models.mysql_db import get_mysql_connection, init_all_tables


def check_mysql_connection() -> bool:
    """MySQL 연결 확인"""
    try:
        conn = get_mysql_connection()
        conn.close()
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("레시피 Agent API 시작!")
    print("="*60)

    rag_system = get_rag_system()
    if rag_system:
        print("RAG 시스템 초기화 완료")

    if check_mysql_connection():
        print("MySQL DB 연결 확인 완료")
        # 모든 테이블 자동 생성
        try:
            init_all_tables()
            print("DB 테이블 자동 생성 완료")
        except Exception as e:
            print(f"DB 테이블 생성 실패: {e}")
    else:
        print("MySQL DB 연결 실패!")

    init_utensils()
    
    try:
        print("📦 랭킹 캐시 로딩 중...")
        await load_today_ranking_cache()
        print("📦 랭킹 캐시 완료")
    except Exception as e:
        print("❌ 랭킹 캐시 로딩 실패")
        print(e)
        traceback.print_exc()

    print("="*60 + "\n")

    yield

    print("\n서버 종료")


app = FastAPI(
    title="레시피 챗봇 Agent API",
    description="RAG + LangGraph 기반 레시피 추천 및 조리모드",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(user_router, prefix="/api/user", tags=["User"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(chat_external_router, prefix="/api/chat-external", tags=["External Chat"])
app.include_router(recipe_router, prefix="/api/recipe", tags=["Recipe"])
app.include_router(cooking_router, prefix="/api/cook", tags=["Cooking"])
app.include_router(mypage_router, prefix="/api/mypage", tags=["MyPage"])
app.include_router(weather_router, prefix="/api/weather", tags=["Weather"])
app.include_router(ranking_router, prefix="/api/rankings", tags=["Ranking"])
app.include_router(voice_router, prefix="/api/voice", tags=["Voice"])

@app.get("/")
async def root():
    return {"message": "Recipe Chatbot API"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_available": get_rag_system() is not None,
        "mysql_available": check_mysql_connection()
    }