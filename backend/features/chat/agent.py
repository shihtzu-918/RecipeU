# backend/features/chat/agent.py
"""
Chat Agent - Adaptive RAG
"""
import os
import time
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# prompts.py에서 프롬프트 import
from .prompts import REWRITE_PROMPT, GRADE_PROMPT, GENERATE_PROMPT
from services.search import get_search_service


# ─────────────────────────────────────────────
# 노드별 타이밍 래퍼
# ─────────────────────────────────────────────
# 누적 타이밍을 저장할 전역 딕셔너리 (요청당 초기화됨)
_node_timings: dict = {}

def timed_node(name: str, fn):
    """노드 함수를 감싸서 실행 시간을 자동 로깅"""
    def wrapper(state: "ChatAgentState") -> "ChatAgentState":
        start = time.time()
        result = fn(state)
        elapsed_ms = (time.time() - start) * 1000
        _node_timings[name] = elapsed_ms
        elapsed_sec = elapsed_ms / 1000
        print(f"  ⏱️  [Node: {name}] {elapsed_sec:.1f}초")
        return result
    return wrapper


class ChatAgentState(TypedDict):
    """Agent 상태"""
    question: str
    original_question: str
    chat_history: List[str]
    documents: List[Document]
    generation: str
    web_search_needed: str
    user_constraints: dict
    constraint_warning: str


def create_chat_agent(rag_system):
    """Chat Agent 생성 - Adaptive RAG + 네이버 검색"""

    search_engine = os.getenv("SEARCH_ENGINE", "serper")
    search_service = get_search_service(search_engine)
    print(f"[Agent] 검색 엔진: {search_engine}")
    
    # ===== 노드 함수 =====

    def check_recipe_relevance(state: ChatAgentState) -> ChatAgentState:
        """관련성 체크 - 원본 질문으로 판단"""
        print("[Agent] 0. 레시피 관련성 체크 중...")
        
        question = state.get("question", state.get("original_question", ""))
        chat_history = state.get("chat_history", [])
        
        recent_context = "\n".join(chat_history[-3:]) if chat_history else ""
        full_context = f"{recent_context}\n사용자: {question}"
        
        relevance_prompt = f"""당신은 레시피 추천 챗봇입니다. 다음 대화가 레시피 추천과 관련이 있는지 판단하세요.

    대화:
    {full_context}

    ✅ 레시피 추천 관련 (RELEVANT):
    - 음식 종류 언급: "짬뽕", "찌개", "파스타", "디저트", "쿠키", "두쫀쿠"
    - 맛/특징 언급: "매운 거", "시원한 거", "차가운 거", "달콤한 거"
    - 상황/니즈: "간단한 요리", "빠른 요리", "야식", "간식"
    - 재료 기반: "김치로", "계란으로", "남은 재료로"
    - 조건 추가: "덜 맵게", "더 달게", "인원 늘려서"
    - 막연한 요청: "뭐 먹을까", "추천해줘", "요리 해볼까"
    - 줄임말/별명: "두쫀쿠", "계란찜", "김볶" 등

    ❌ 레시피 추천 무관 (NOT_RELEVANT):
    - 날씨: "날씨가 좋네", "비 오네"
    - 일반 상식: "보냉팩 분리수거", "칼 쓰는 법"
    - 뉴스/시사: "오늘 뉴스", "주식"
    - 요리와 완전 무관: "영화 추천", "여행지"

    중요: 음식/맛/식사와 조금이라도 관련 있으면 RELEVANT!

    답변 (한 단어만):"""
        
        try:
            from langchain_core.messages import HumanMessage
            result = rag_system.chat_model.invoke([HumanMessage(content=relevance_prompt)])
            decision = result.content.strip().upper()
            
            print(f"   LLM 판단: {decision}")
            
            if "NOT" in decision and "RELEVANT" in decision:
                print(f"   → 레시피 생성 무관")
                return {
                    "generation": "NOT_RECIPE_RELATED",
                    "documents": []
                }
            else:
                print(f"   → 레시피 생성 관련")
                return {}
                
        except Exception as e:
            print(f"   관련성 체크 실패: {e}")
            return {}
    
    def rewrite_query(state: ChatAgentState) -> ChatAgentState:
        """1. 쿼리 재작성"""
        print("[Agent] 1. 쿼리 재작성 중...")
        
        question = state["question"]
        history = state.get("chat_history", [])
        
        formatted_history = "\n".join(history[-5:]) if isinstance(history, list) else str(history)
        
        try:
            chain = REWRITE_PROMPT | rag_system.chat_model | StrOutputParser()
            better_question = chain.invoke({
                "history": formatted_history,
                "question": question
            })
            
            print(f"   원본: {question}")
            print(f"   재작성: {better_question}")
            
            return {
                "question": better_question,
                "original_question": question
            }
            
        except Exception as e:
            print(f"   재작성 실패: {e}")
            return {
                "question": question,
                "original_question": question
            }
    
    def retrieve(state: ChatAgentState) -> ChatAgentState:
        """2. RAG 검색 (Reranker 사용)"""
        print("[Agent] 2. RAG 검색 중...")
        
        question = state["question"]
        
        # use_rerank=None -> RAG 시스템 설정(USE_RERANKER) 따름
        results = rag_system.search_recipes(question, k=3, use_rerank=None)
        
        documents = [
            Document(
                page_content=doc.get("content", ""),
                metadata={
                    "title": doc.get("title", ""),
                    "cook_time": doc.get("cook_time", ""),
                    "level": doc.get("level", "")
                }
            )
            for doc in results
        ]
        
        print(f"   검색 결과: {len(documents)}개")
        for i, doc in enumerate(documents[:3], 1):
            print(f"   {i}. {doc.metadata.get('title', '')[:40]}...")
        
        return {"documents": documents}
    
    def check_constraints(state: ChatAgentState) -> ChatAgentState:
        """2.5. 제약 조건 체크 (알레르기, 비선호 음식)"""
        print("[Agent] 2.5. 제약 조건 체크 중...")
        
        question = state["question"]
        user_constraints = state.get("user_constraints", {})
        
        if not user_constraints:
            print("   제약 조건 없음 → 스킵")
            return {"constraint_warning": ""}
        
        dislikes = user_constraints.get("dislikes", [])
        allergies = user_constraints.get("allergies", [])
        
        question_lower = question.lower()
        warning_parts = []

        for allergy in allergies:
            if allergy.lower() in question_lower:
                warning_parts.append(f"**{allergy}**는 알레르기 재료입니다!")
        
        for dislike in dislikes:
            if dislike.lower() in question_lower:
                warning_parts.append(f"**{dislike}**는 싫어하는 음식입니다.")
    
        if warning_parts:
            warning_msg = "\n".join(warning_parts)
            print(f"   제약 조건 위반 감지!")
            print(f"   {warning_msg}")
            return {"constraint_warning": warning_msg}
        else:
            print("   제약 조건 통과")
            return {"constraint_warning": ""}

    def grade_documents(state: ChatAgentState) -> ChatAgentState:
        """3. 문서 관련성 평가"""
        print("[Agent] 3. 관련성 평가 중...")
        
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            print("   문서 없음 → 웹 검색")
            return {"web_search_needed": "yes"}
        
        try:
            question_lower = question.lower()
            
            found_exact_match = False
            for doc in documents[:3]:
                title = doc.metadata.get("title", "").lower()
                if question_lower in title or any(
                    word in title 
                    for word in question_lower.split() 
                    if len(word) > 1
                ):
                    found_exact_match = True
                    break
            
            if not found_exact_match:
                print("   제목 매칭 실패 → 웹 검색")
                return {"web_search_needed": "yes"}
            
            context_text = "\n".join([
                f"- {doc.page_content[:200]}"
                for doc in documents[:3]
            ])
            
            chain = GRADE_PROMPT | rag_system.chat_model | StrOutputParser()
            score = chain.invoke({
                "question": question,
                "context": context_text
            })
            
            print(f"   평가: {score}")
            
            if "yes" in score.lower():
                print("   DB 충분 → 생성")
                return {"web_search_needed": "no"}
            else:
                print("   DB 부족 → 웹 검색")
                return {"web_search_needed": "yes"}
                
        except Exception as e:
            print(f"   평가 실패: {e}")
            return {"web_search_needed": "yes"}
    
    def web_search(state: ChatAgentState) -> ChatAgentState:
        """4. 웹 검색"""
        print("[Agent] 4. 웹 검색 실행 중...")
        
        question = state["question"]
        documents = search_service.search(query=question, max_results=5)
        
        for i, doc in enumerate(documents, 1):
            print(f"\n   [검색 결과 {i}]")
            print(f"   제목: {doc.metadata.get('title', '')}")
            print(f"   내용: {doc.page_content[:200]}...")
        
        return {"documents": documents}

    def generate(state: ChatAgentState) -> ChatAgentState:
        """5. 답변 생성 (이미지 제거)"""
        print("[Agent] 5. 답변 생성 중...")
        
        question = state["original_question"]
        documents = state["documents"]
        history = state.get("chat_history", [])
        constraint_warning = state.get("constraint_warning", "")
        user_constraints = state.get("user_constraints", {})
        
        formatted_history = "\n".join(history[-10:]) if isinstance(history, list) else str(history)
        
        context_text = "\n\n".join([
            doc.page_content[:800]
            for doc in documents
        ])
        
        if constraint_warning:
            try:
                alt_prompt = f"""{constraint_warning}

    그래도 레시피를 원하시나요? 
    아니면 비슷한 다른 재료로 대체할까요?

    답변:"""
                
                from langchain_core.messages import HumanMessage
                result = rag_system.chat_model.invoke([HumanMessage(content=alt_prompt)])
                answer = f"{constraint_warning}\n\n{result.content.strip()}"
                
                return {"generation": answer}
                
            except Exception as e:
                print(f"   경고 생성 실패: {e}")
                return {"generation": f"{constraint_warning}\n\n다른 요리를 추천해드릴까요?"}
        
        try:
            if user_constraints:
                allergies = user_constraints.get("allergies", [])
                dislikes = user_constraints.get("dislikes", [])
                
                constraints_text = ""
                if allergies:
                    constraints_text += f"\n알레르기 재료 (절대 사용 금지): {', '.join(allergies)}"
                if dislikes:
                    constraints_text += f"\n비선호 음식 (피해야 함): {', '.join(dislikes)}"
                
                enhanced_context = f"""{constraints_text}

    {context_text}"""
            else:
                enhanced_context = context_text
            
            chain = GENERATE_PROMPT | rag_system.chat_model | StrOutputParser()
            answer = chain.invoke({
                "context": enhanced_context,
                "question": question,
                "history": formatted_history
            })
            
            print(f"   생성 완료: {answer[:50]}...")
            return {"generation": answer}
            
        except Exception as e:
            print(f"   생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return {"generation": "답변 생성에 실패했습니다."}

    # ===== 그래프 구성 =====
    
    def decide_to_generate(state: ChatAgentState) -> Literal["web_search", "generate"]:
        """grade 노드 이후 분기 결정"""
        if state.get("web_search_needed") == "yes":
            return "web_search"
        else:
            return "generate"
    
    workflow = StateGraph(ChatAgentState)
    
    # ── 모든 노드를 timed_node로 감싸기 ──
    workflow.add_node("check_relevance",  timed_node("check_relevance",  check_recipe_relevance))
    workflow.add_node("rewrite",          timed_node("rewrite",          rewrite_query))
    workflow.add_node("retrieve",         timed_node("retrieve",         retrieve))
    workflow.add_node("check_constraints",timed_node("check_constraints",check_constraints))
    workflow.add_node("grade",            timed_node("grade",            grade_documents))
    workflow.add_node("web_search",       timed_node("web_search",       web_search))
    workflow.add_node("generate",         timed_node("generate",         generate))
    
    workflow.set_entry_point("check_relevance")
    
    workflow.add_conditional_edges(
        "check_relevance",
        lambda state: "end" if state.get("generation") == "NOT_RECIPE_RELATED" else "rewrite",
        {"end": END, "rewrite": "rewrite"}
    )
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "check_constraints")
    workflow.add_edge("check_constraints", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {"web_search": "web_search", "generate": "generate"}
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    compiled = workflow.compile()
    
    print("[Agent] Adaptive RAG Agent 생성 완료 (네이버 검색 API)")
    return compiled