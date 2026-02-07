# backend/app/config.py
"""
설정 및 환경변수 관리
"""
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional


class Settings(BaseSettings):
    # API 설정
    APP_NAME: str = "Recipe Agent API"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://localhost:5174", 
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174", 
    "http://198.18.16.237:5173",
    "https://recipeu.site",
    "https://www.recipeu.site"
    ]
    
    # CLOVA Studio
    CLOVASTUDIO_API_KEY: str = ""
    CLOVA_INVOKE_URL: str = ""
    CLOVA_SECRET_KEY: str = ""
    CLOVA_TTS_CLIENT_ID: str = ""
    CLOVA_TTS_CLIENT_SECRET: str = ""

    # Clova Speech STT (Voice 모듈용)
    CLOVA_STT_INVOKE_URL: str = ""
    CLOVA_STT_SECRET_KEY: str = ""

    # RunPod API (LLM/TTS 서버)
    RECIPEU_API_KEY: str = ""

    # NAVER
    NAVER_CLIENT_ID: Optional[str] = None
    NAVER_CLIENT_SECRET: Optional[str] = None

    # Serper.dev (웹 검색)
    SERPER_API_KEY: Optional[str] = None
    
    # Milvus
    MILVUS_HOST: str = ""
    MILVUS_PORT: str = "19530"
    COLLECTION_NAME: str = "recipe_docs"

    USE_RERANKER: bool = False

    # MySQL (Naver Cloud)
    MYSQL_HOST: str = ""
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()