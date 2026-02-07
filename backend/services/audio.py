# services/audio.py
"""
음성 처리 서비스 - 기존 audioagent.py 내용
"""
import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI


class AudioAgent:
    """음성 처리 Agent (STT/TTS)"""
    
    def __init__(self):
        load_dotenv()
        
        self.stt_invoke_url = os.getenv("CLOVA_INVOKE_URL")
        self.stt_secret_key = os.getenv("CLOVA_SECRET_KEY")
        self.studio_secret = os.getenv("CLOVASTUDIO_API_KEY")
        self.tts_client_id = os.getenv("CLOVA_TTS_CLIENT_ID")
        self.tts_client_secret = os.getenv("CLOVA_TTS_CLIENT_SECRET")
        
        self.tts_url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
        self.llm_base_url = "https://clovastudio.stream.ntruss.com/v1/openai"
        
        self._validate_keys()
        
        self.llm_client = OpenAI(
            api_key=self.studio_secret,
            base_url=self.llm_base_url
        )
    
    def _validate_keys(self):
        keys = [
            self.stt_invoke_url,
            self.stt_secret_key,
            self.studio_secret,
            self.tts_client_id,
            self.tts_client_secret
        ]
        if not all(keys):
            raise ValueError("환경 변수에 필요한 API 키가 누락되었습니다.")
    
    def stt(self, file_path: str):
        """음성을 텍스트로 변환 (STT)"""
        print(f"STT 변환 시작: {file_path}")
        request_url = f"{self.stt_invoke_url}/recognizer/upload"
        headers = {'X-CLOVASPEECH-API-KEY': self.stt_secret_key}
        
        params = {
            'language': 'ko-KR',
            'completion': 'sync',
            'callback': '',
            'userdata': {'test': '1'},
            'wordAlignment': True,
            'fullText': True,
            'forbiddens': '',
            'boostings': '',
            'diarization': {
                'enable': False,
                'speakerCountMin': None,
                'speakerCountMax': None,
            },
        }

        try:
            with open(file_path, 'rb') as f:
                files = {
                    'media': f,
                    'params': (None, json.dumps(params), 'application/json')
                }
                response = requests.post(request_url, headers=headers, files=files)
                response.raise_for_status()
                result = response.json()
                print(f">> STT 완료: {result.get('text', '')[:30]}...")
                return result.get('text', '')
        except Exception as e:
            print(f"STT 오류: {e}")
            return None