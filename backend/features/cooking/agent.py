# features/cooking/agent.py
"""
Cooking Agent (LangGraph)
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
import operator

from features.cooking.schemas import CookingAgentState
from utils.intent import detect_intent, Intent


class CookingAgent:
    """조리모드 Agent - 음성 안내 + 단계 관리"""
    
    def __init__(self, rag_system, session):
        self.rag = rag_system
        self.session = session
        self.agent = self._create_agent()
        self.state = None
    
    def _create_agent(self):
        """조리모드 Agent 그래프 생성"""
        
        def detect_intent_node(state: CookingAgentState) -> CookingAgentState:
            """의도 감지"""
            user_input = state["user_input"]
            intent = detect_intent(user_input)
            
            # 다음 액션 결정
            if intent == Intent.NEXT:
                next_action = "next"
            elif intent == Intent.PREV:
                next_action = "prev"
            elif intent in [Intent.SUB_ING, Intent.SUB_TOOL]:
                next_action = "substitute"
            elif intent == Intent.FAILURE:
                next_action = "failure"
            else:
                next_action = "continue"
            
            return {
                "intent": intent,
                "next_action": next_action
            }
        
        def handle_navigation(state: CookingAgentState) -> CookingAgentState:
            """단계 이동 (다음/이전)"""
            action = state["next_action"]
            
            if action == "next":
                msg, new_step = self.session._go_next()
            else:  # prev
                msg, new_step = self.session._go_prev()
            
            # TTS 생성
            audio_path = self.session.generate_tts(msg)
            
            return {
                "response": msg,
                "current_step": new_step,
                "audio_path": audio_path,
                "history": [{"role": "assistant", "content": msg}],
                "next_action": "end"
            }
        
        def handle_substitute(state: CookingAgentState) -> CookingAgentState:
            """재료/도구 대체"""
            user_input = state["user_input"]
            intent = state["intent"]
            
            mode = "ingredient" if intent == Intent.SUB_ING else "tool"
            msg, _ = self.session._handle_substitute(user_input, mode=mode)
            
            # TTS 생성
            audio_path = self.session.generate_tts(msg)
            
            return {
                "response": msg,
                "audio_path": audio_path,
                "history": [{"role": "assistant", "content": msg}],
                "next_action": "end"
            }
        
        def handle_failure(state: CookingAgentState) -> CookingAgentState:
            """조리 실패 대응"""
            user_input = state["user_input"]
            msg, _ = self.session._handle_failure(user_input)
            
            # TTS 생성
            audio_path = self.session.generate_tts(msg)
            
            return {
                "response": msg,
                "audio_path": audio_path,
                "history": [{"role": "assistant", "content": msg}],
                "next_action": "end"
            }
        
        def handle_general(state: CookingAgentState) -> CookingAgentState:
            """일반 질문 처리"""
            msg = (
                "지금은 조리 중이라 다음 기능만 지원해요: "
                "다음/이전 단계, 재료 대체, 도구 대체, 실패 대응."
            )
            
            audio_path = self.session.generate_tts(msg)
            
            return {
                "response": msg,
                "audio_path": audio_path,
                "history": [{"role": "assistant", "content": msg}],
                "next_action": "end"
            }
        
        def route_action(state: CookingAgentState) -> str:
            """다음 노드 라우팅"""
            action = state.get("next_action", "continue")
            
            if action in ["next", "prev"]:
                return "navigate"
            elif action == "substitute":
                return "substitute"
            elif action == "failure":
                return "failure"
            elif action == "continue":
                return "general"
            else:
                return "end"
        
        # 그래프 구성
        workflow = StateGraph(CookingAgentState)
        
        workflow.add_node("detect_intent", detect_intent_node)
        workflow.add_node("navigate", handle_navigation)
        workflow.add_node("substitute", handle_substitute)
        workflow.add_node("failure", handle_failure)
        workflow.add_node("general", handle_general)
        
        workflow.set_entry_point("detect_intent")
        workflow.add_conditional_edges("detect_intent", route_action)
        workflow.add_edge("navigate", END)
        workflow.add_edge("substitute", END)
        workflow.add_edge("failure", END)
        workflow.add_edge("general", END)
        
        return workflow.compile()
    
    def set_recipe(self, recipe: dict):
        """레시피 설정"""
        self.session.set_recipe(recipe)
        
        steps = recipe.get("steps", [])
        self.state = {
            "recipe": recipe,
            "current_step": 0,
            "total_steps": len(steps),
            "history": [],
            "user_input": "",
            "intent": "",
            "response": "",
            "audio_path": "",
            "next_action": "continue"
        }
    
    async def handle_input(self, user_input: str) -> dict:
        """사용자 입력 처리"""
        if not self.state:
            return {
                "error": "레시피가 설정되지 않았습니다."
            }
        
        # 상태 업데이트
        self.state["user_input"] = user_input
        self.state["history"].append({"role": "user", "content": user_input})
        
        # Agent 실행
        final_state = await self.agent.ainvoke(self.state)
        
        # 상태 업데이트
        self.state = final_state
        
        return {
            "response": final_state["response"],
            "audio_path": final_state["audio_path"],
            "current_step": final_state["current_step"],
            "total_steps": final_state["total_steps"]
        }
    
    async def handle_audio(self, audio_path: str) -> dict:
        """음성 입력 처리 (STT → Agent)"""
        # STT
        user_text = self.session.speech.stt(audio_path) or ""
        
        if not user_text.strip():
            msg = "음성을 인식하지 못했어요."
            tts_path = self.session.generate_tts(msg)
            return {
                "response": msg,
                "audio_path": tts_path,
                "current_step": self.state["current_step"],
                "total_steps": self.state["total_steps"]
            }
        
        # Agent 처리
        return await self.handle_input(user_text)