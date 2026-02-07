# features/cooking/session.py
"""
CookingSession - 기존 cooking_workflow.py 내용
"""
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.audio import AudioAgent


@dataclass
class CookingSession:
    """조리 세션"""
    rag: Any
    speech: Optional[AudioAgent] = None

    recipe_json: Optional[Dict[str, Any]] = None
    step_index: int = 0

    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.speech is None:
            self.speech = AudioAgent()

    def set_recipe(self, recipe_json: Dict[str, Any]):
        """레시피 설정"""
        self.recipe_json = recipe_json
        self.step_index = 0
        self.history.clear()

    def handle_text(self, user_text: str) -> Tuple[str, int]:
        """텍스트 입력 처리"""
        from utils.intent import detect_intent, Intent
        
        if not self.recipe_json:
            return "레시피가 아직 설정되지 않았어요.", self.step_index

        intent = detect_intent(user_text)
        self.history.append({"role": "user", "content": user_text})

        if intent == Intent.NEXT:
            return self._go_next()
        if intent == Intent.PREV:
            return self._go_prev()
        if intent == Intent.SUB_ING:
            return self._handle_substitute(user_text, mode="ingredient")
        if intent == Intent.SUB_TOOL:
            return self._handle_substitute(user_text, mode="tool")
        if intent == Intent.FAILURE:
            return self._handle_failure(user_text)

        msg = "지금은 조리 중이라 다음 기능만 지원해요."
        self.history.append({"role": "assistant", "content": msg})
        return msg, self.step_index

    def _go_next(self) -> Tuple[str, int]:
        """다음 단계"""
        steps = self.recipe_json.get("steps") or []
        if self.step_index >= len(steps) - 1:
            msg = "마지막 단계예요."
            self.history.append({"role": "assistant", "content": msg})
            return msg, self.step_index
        
        self.step_index += 1
        step = steps[self.step_index]
        msg = f"{step.get('no', self.step_index+1)}단계: {step.get('desc','')}"
        self.history.append({"role": "assistant", "content": msg})
        return msg, self.step_index

    def _go_prev(self) -> Tuple[str, int]:
        """이전 단계"""
        if self.step_index <= 0:
            steps = self.recipe_json.get("steps") or []
            step = steps[0] if steps else {}
            msg = f"이미 1단계예요. {step.get('desc','')}"
            self.history.append({"role": "assistant", "content": msg})
            return msg, self.step_index
        
        self.step_index -= 1
        steps = self.recipe_json.get("steps") or []
        step = steps[self.step_index]
        msg = f"{step.get('no', self.step_index+1)}단계: {step.get('desc','')}"
        self.history.append({"role": "assistant", "content": msg})
        return msg, self.step_index

    def _handle_substitute(self, user_text: str, mode: str) -> Tuple[str, int]:
        """대체재료/도구"""
        steps = self.recipe_json.get("steps") or []
        step = steps[self.step_index] if self.step_index < len(steps) else {}
        title = self.recipe_json.get("title", "이 요리")
        step_desc = step.get("desc", "")

        prompt = f"""
요리: {title}
현재 단계: {self.step_index+1}단계
단계 설명: {step_desc}
사용자: {user_text}

너는 조리 중 도우미야. 한국어로 아주 짧게(3~5문장) 답해.
"""
        if mode == "ingredient":
            prompt += "대체재료 2~3개와 주의사항을 알려줘."
        else:
            prompt += "대체도구/대체조리법 2~3개와 안전 주의사항을 알려줘."

        try:
            resp = self.speech.llm_client.chat.completions.create(
                model="HCX-005",
                messages=[
                    {"role": "system", "content": "너는 조리 어시스턴트야."},
                    {"role": "user", "content": prompt},
                ],
            )
            msg = resp.choices[0].message.content.strip()
        except Exception as e:
            msg = f"대체 제안 생성 중 오류: {e}"

        self.history.append({"role": "assistant", "content": msg})
        return msg, self.step_index

    def _handle_failure(self, user_text: str) -> Tuple[str, int]:
        """조리 실패 대응"""
        steps = self.recipe_json.get("steps") or []
        step = steps[self.step_index] if self.step_index < len(steps) else {}
        title = self.recipe_json.get("title", "이 요리")
        step_desc = step.get("desc", "")

        prompt = f"""
요리: {title}
현재 단계: {self.step_index+1}단계
단계 설명: {step_desc}
사용자: {user_text}

너는 조리 사고 대응 전문가야. 한국어로 4~6문장.
1) 응급조치 2~3개
2) 복구 방법 1~2개
3) 예방 팁 1개
"""

        try:
            resp = self.speech.llm_client.chat.completions.create(
                model="HCX-005",
                messages=[
                    {"role": "system", "content": "너는 요리 사고 대응 코치야."},
                    {"role": "user", "content": prompt},
                ],
            )
            msg = resp.choices[0].message.content.strip()
        except Exception as e:
            msg = f"실패 대응 생성 중 오류: {e}"

        self.history.append({"role": "assistant", "content": msg})
        return msg, self.step_index

    def generate_tts(self, text: str, voice: str = "nara_call") -> str:
        """TTS 생성"""
        if not self.speech:
            raise RuntimeError("SpeechAgent가 초기화되지 않았습니다.")

        out_dir = os.path.join(tempfile.gettempdir(), "cook_tts")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"tts_{os.getpid()}_{len(self.history)}.wav")

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.speech.tts_client_id,
            "X-NCP-APIGW-API-KEY": self.speech.tts_client_secret,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"speaker": voice, "speed": "0", "text": text}

        import requests
        resp = requests.post(self.speech.tts_url, headers=headers, data=data)
        if resp.status_code != 200:
            raise RuntimeError(f"TTS API 오류: {resp.status_code}")
        
        with open(out_path, "wb") as f:
            f.write(resp.content)