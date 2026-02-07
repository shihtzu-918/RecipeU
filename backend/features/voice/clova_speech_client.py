# features/voice/clova_speech_client.py
"""
Naver Clova Speech (STT) 클라이언트
- 음성 바이트를 받아 텍스트로 변환
- httpx 비동기 클라이언트 사용
"""
import json
import httpx


class ClovaSpeechClient:
    def __init__(self, invoke_url: str, secret_key: str):
        self.invoke_url = invoke_url.rstrip("/")
        self.secret_key = secret_key

    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        음성 바이트 → 텍스트 (Clova Speech API upload 방식)
        Returns: 인식된 텍스트 (빈 문자열이면 인식 실패)
        """
        request_url = f"{self.invoke_url}/recognizer/upload"

        params = {
            "language": "ko-KR",
            "completion": "sync",
            "callback": "",
            "userdata": {"id": "voice_pipeline"},
            "wordAlignment": False,
            "fullText": True,
            "diarization": {"enable": False},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                request_url,
                headers={"X-CLOVASPEECH-API-KEY": self.secret_key},
                files={
                    "media": ("audio.webm", audio_bytes, "application/octet-stream"),
                    "params": (None, json.dumps(params, ensure_ascii=False), "application/json"),
                },
            )
            response.raise_for_status()
            result = response.json()

            print(f"[ClovaSpeech] 응답: {result}")

            text = result.get("text", "").strip()
            return text
