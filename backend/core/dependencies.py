# core/dependencies.py
"""
FastAPI 의존성 관리
"""
from functools import lru_cache
from typing import Optional

from services.rag import RecipeRAGLangChain
from app.config import settings


# 싱글톤 인스턴스
_rag_system: Optional[RecipeRAGLangChain] = None


@lru_cache()
def get_rag_system() -> Optional[RecipeRAGLangChain]:
    """RAG 시스템 싱글톤"""
    global _rag_system

    if _rag_system is None:
        if not settings.CLOVASTUDIO_API_KEY:
            print("CLOVASTUDIO_API_KEY 환경변수가 설정되지 않았습니다.")
            return None

        try:
            _rag_system = RecipeRAGLangChain(
                milvus_host=settings.MILVUS_HOST,
                milvus_port=settings.MILVUS_PORT,
                collection_name=settings.COLLECTION_NAME,
                use_reranker=settings.USE_RERANKER,
                temperature=0.01,
                max_tokens=2000,
            )
        except Exception as e:
            print(f"RAG 초기화 실패: {e}")
            return None

    return _rag_system