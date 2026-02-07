# run.py
"""
서버 실행 스크립트
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,  # 개발 모드
        log_level="info"
    )