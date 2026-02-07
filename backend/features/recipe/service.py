# backend/features/recipe/service.py
"""
Recipe 비즈니스 로직
"""
import os
import re
from pymongo import MongoClient
from typing import List, Dict, Any
from .prompts import RECIPE_QUERY_EXTRACTION_PROMPT, RECIPE_GENERATION_PROMPT


class RecipeService:
    def __init__(self, rag_system, recipe_db, user_profile=None):
        mongo_uri = os.getenv("MONGO_URI", "mongodb://root:RootPassword123@136.113.251.237:27017/admin")
        self.mongo_client = MongoClient(mongo_uri)
        self.recipe_db = self.mongo_client["recipe_db"]
        self.recipes_collection = self.recipe_db["recipes"]
        self.rag = rag_system
        self.db = recipe_db
        self.user_profile = user_profile or {}
    
    async def generate_recipe(
        self, 
        chat_history: List[Dict[str, str]],
        member_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """상세 레시피 생성 (대화 기반) + 이미지 URL"""
        
        print(f"[RecipeService] 레시피 생성 시작")
        print(f"[RecipeService] 대화 개수: {len(chat_history)}")
        print(f"[RecipeService] 가족 정보: {member_info}")
        
        # 1. LLM으로 대화 분석 + 검색 쿼리 생성
        search_query = self._extract_search_query_with_llm(chat_history, member_info)
        
        print(f"[RecipeService] 생성된 검색 쿼리: {search_query}")
        
        # 2. RAG 검색
        retrieved_docs = self.rag.search_recipes(search_query, k=3, use_rerank=False)
        
        print(f"[RecipeService] RAG 검색 결과: {len(retrieved_docs)}개")
        
        # 웹 검색 여부 판단
        from_web_search = not retrieved_docs or len(retrieved_docs) == 0
        
        # 3. 알레르기/비선호 필터링
        filtered_docs = self._filter_by_constraints(retrieved_docs, member_info)
        
        print(f"[RecipeService] 필터링 후: {len(filtered_docs)}개")
        
        # 4. LLM으로 최종 레시피 생성
        recipe_json = self._generate_final_recipe_with_llm(
            chat_history=chat_history,
            member_info=member_info,
            context_docs=filtered_docs
        )
        
        print(f"[RecipeService] 레시피 생성 완료: {recipe_json.get('title')}")
        
        # 5. 이미지 찾기
        recipe_title = recipe_json.get('title', '')
        best_image = ""
        
        if from_web_search:
            # 웹 검색이면 기본 이미지
            print(f"[RecipeService] 웹 검색 레시피 → 기본 이미지 사용")
            best_image = 'https://kr.object.ncloudstorage.com/recipu-bucket/assets/default_img.webp'
        else:
            # RAG 검색이면 MongoDB에서 찾기 (미사여구 제거)
            if recipe_title:
                best_image = self._find_image_by_title(recipe_title)
            
            # MongoDB에서도 못 찾으면 원본 검색 결과에서
            if not best_image:
                print(f"[RecipeService] 제목 검색 실패 → 원본 검색 결과 사용")
                best_image = self._get_best_image(filtered_docs)
        
        print(f"[RecipeService] 선택된 이미지: {best_image or '기본 이미지'}")
        
        # 6. 이미지 URL 추가
        recipe_json['image'] = best_image
        recipe_json['img_url'] = best_image
        
        # 7. 인원수 설정
        servings = len(member_info.get('names', [])) if member_info and member_info.get('names') else 1
        if 'servings' not in recipe_json or not recipe_json['servings']:
            recipe_json['servings'] = f"{servings}인분"
        
        print(f"[RecipeService] 최종 레시피: {recipe_json.get('title')}")
        print(f"[RecipeService] 인원수: {recipe_json['servings']}")
        print(f"[RecipeService] 이미지: {recipe_json.get('image', 'None')[:60]}...")
        
        return recipe_json
    
    def _find_image_by_title(self, title: str) -> str:
        """
        MongoDB에서 제목으로 이미지 직접 검색
        미사여구 제거하고 핵심 요리명만으로 검색
        """
        try:
            # ✅ 미사여구 제거
            clean_title = title
            clean_title = re.sub(r'\([^)]*\)', '', clean_title)  # 괄호 안 내용 제거
            clean_title = re.sub(r'\[[^\]]*\]', '', clean_title)  # 대괄호 안 내용 제거
            clean_title = re.sub(r'[~!@#$%^&*()_+|<>?:{}]', '', clean_title)  # 특수문자 제거
            clean_title = re.sub(r'\s+', ' ', clean_title)  # 연속 공백 제거
            clean_title = clean_title.strip()
            
            print(f"[RecipeService] 정제된 제목: '{title}' → '{clean_title}'")
            
            # 정확한 매칭 시도 (정제된 제목)
            recipe = self.recipes_collection.find_one(
                {"title": {"$regex": re.escape(clean_title), "$options": "i"}}, 
                {"image": 1, "recipe_id": 1, "title": 1, "_id": 0}
            )
            
            if recipe and "image" in recipe:
                image_url = recipe["image"]
                matched_title = recipe.get("title", "")
                print(f"[RecipeService] MongoDB 제목 매칭: {matched_title}")
                print(f"[RecipeService] 이미지: {image_url[:60]}...")
                return image_url
            
            # 부분 매칭 시도 (키워드)
            keywords = [word for word in clean_title.split() if len(word) > 1]
            
            if keywords:
                # 키워드로 검색
                recipe = self.recipes_collection.find_one(
                    {"title": {"$regex": "|".join(keywords), "$options": "i"}},
                    {"image": 1, "recipe_id": 1, "title": 1, "_id": 0}
                )
                
                if recipe and "image" in recipe:
                    image_url = recipe["image"]
                    matched_title = recipe.get("title", "")
                    print(f"[RecipeService] MongoDB 키워드 매칭: {matched_title}")
                    print(f"[RecipeService] 이미지: {image_url[:60]}...")
                    return image_url
            
            print(f"[RecipeService] MongoDB에서 '{clean_title}' 찾지 못함")
            return ""
            
        except Exception as e:
            print(f"[RecipeService] MongoDB 제목 검색 실패: {e}")
            return ""
    
    def _get_image_from_mongo(self, recipe_id: str) -> str:
        """MongoDB에서 레시피 이미지 URL 가져오기"""
        try:
            recipe = self.recipes_collection.find_one(
                {"recipe_id": recipe_id},
                {"image": 1, "_id": 0}
            )
            
            if recipe and "image" in recipe:
                image_url = recipe["image"]
                print(f"[RecipeService] MongoDB 이미지: {image_url[:50]}...")
                return image_url
            else:
                print(f"[RecipeService] MongoDB에 이미지 없음: recipe_id={recipe_id}")
                return ""
                
        except Exception as e:
            print(f"[RecipeService] MongoDB 이미지 조회 실패: {e}")
            return ""
    
    def _get_best_image(self, filtered_docs: List[Dict]) -> str:
        """
        필터링된 레시피 중 이미지 선택
        제목 검색 실패 후 여기 온 거면 그냥 기본 이미지 사용
        """
        print("[RecipeService] 제목 검색 실패 → 기본 이미지 사용")
        return "https://kr.object.ncloudstorage.com/recipu-bucket/assets/default_img.webp"
    
    def _extract_search_query_with_llm(
        self, 
        chat_history: List[Dict],
        member_info: Dict
    ) -> str:
        """LLM으로 검색 쿼리 추출"""
        
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[-10:]
        ])
        
        servings = len(member_info.get('names', [])) if member_info else 1
        allergies = ', '.join(member_info.get('allergies', [])) if member_info else '없음'
        dislikes = ', '.join(member_info.get('dislikes', [])) if member_info else '없음'
        
        # 프롬프트 사용
        prompt = RECIPE_QUERY_EXTRACTION_PROMPT.format(
            conversation=conversation,
            servings=servings,
            allergies=allergies,
            dislikes=dislikes
        )
        
        from langchain_naver import ChatClovaX
        llm = ChatClovaX(model="HCX-003", temperature=0.1, max_tokens=50)
        
        try:
            result = llm.invoke(prompt)
            query = result.content.strip()
            print(f"[RecipeService] LLM 추출 쿼리: {query}")
            return query
        except Exception as e:
            print(f"[RecipeService] 쿼리 추출 실패: {e}")
            return self._simple_keyword_extraction(chat_history)
    
    def _simple_keyword_extraction(self, chat_history: List[Dict]) -> str:
        """간단한 키워드 추출 (Fallback)"""
        food_keywords = []
        
        for msg in chat_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                if any(k in content for k in ['찌개', '국', '탕', '볶음', '구이', '조림']):
                    words = content.split()
                    food_keywords.extend([w for w in words if len(w) > 1])
        
        return ' '.join(food_keywords[:5]) if food_keywords else "한식 요리"
    
    def _filter_by_constraints(
        self,
        recipes: List[Dict],
        member_info: Dict
    ) -> List[Dict]:
        """알레르기/비선호 필터링"""
        
        if not member_info:
            return recipes[:5]
        
        filtered = []
        
        for recipe in recipes:
            content = recipe.get("content", "").lower()
            
            # 알레르기 체크
            if member_info.get("allergies"):
                has_allergen = any(
                    allergen.lower() in content 
                    for allergen in member_info["allergies"]
                )
                if has_allergen:
                    continue
            
            # 비선호 재료 체크
            if member_info.get("dislikes"):
                has_dislike = any(
                    dislike.lower() in content 
                    for dislike in member_info["dislikes"]
                )
                if has_dislike:
                    continue
            
            filtered.append(recipe)
            
            if len(filtered) >= 5:
                break
        
        if len(filtered) < 3:
            return recipes[:3]
        
        return filtered
    
    def _generate_final_recipe_with_llm(
        self,
        chat_history: List[Dict],
        member_info: Dict,
        context_docs: List[Dict]
    ) -> Dict:
        """LLM으로 최종 레시피 JSON 생성"""
        
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history
        ])
        
        context_text = "\n\n".join([
            f"[레시피 {i+1}] {doc.get('title')}\n{doc.get('content', '')[:800]}"
            for i, doc in enumerate(context_docs[:5])
        ])
        
        servings = len(member_info.get('names', [])) if member_info else 1
        allergies = ', '.join(member_info.get('allergies', [])) if member_info else '없음'
        dislikes = ', '.join(member_info.get('dislikes', [])) if member_info else '없음'
        tools = ', '.join(member_info.get('tools', [])) if member_info else '모든 도구'
        
        # 프롬프트 사용
        prompt = RECIPE_GENERATION_PROMPT.format(
            conversation=conversation,
            servings=servings,
            allergies=allergies,
            dislikes=dislikes,
            tools=tools,
            context=context_text
        )
        
        from langchain_naver import ChatClovaX
        llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=2000)
        
        try:
            result = llm.invoke(prompt)
            response_text = result.content.strip()
            
            # JSON 추출
            import json
            
            # 마크다운 코드 블록 제거
            response_text = re.sub(r'```json\s*|\s*```', '', response_text)
            
            recipe_json = json.loads(response_text)
            
            print(f"[RecipeService] 레시피 생성 성공: {recipe_json.get('title')}")
            return recipe_json
            
        except json.JSONDecodeError as e:
            print(f"[RecipeService] JSON 파싱 실패: {e}")
            print(f"[RecipeService] 응답: {response_text[:200]}")
            
            # Fallback
            return {
                "title": "추천 레시피",
                "intro": "레시피 생성 중 오류가 발생했습니다.",
                "cook_time": "30분",
                "level": "중급",
                "servings": f"{servings}인분",
                "ingredients": [],
                "steps": [],
                "tips": []
            }
        
        except Exception as e:
            print(f"[RecipeService] 레시피 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise