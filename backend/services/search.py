# backend/services/search.py
"""
웹 검색 서비스 (네이버, 구글 등)
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import os
import requests
from langchain_core.documents import Document


class WebSearchService(ABC):
    """웹 검색 추상 클래스"""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """검색 실행"""
        pass


class NaverBlogSearch(WebSearchService):
    """네이버 블로그 검색"""
    
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """네이버 블로그 검색"""
        print(f"[NaverSearch] 검색 쿼리: {query}")
        
        if not self.client_id or not self.client_secret:
            print("   네이버 API 키 없음")
            return [Document(
                page_content="네이버 API 키가 필요합니다.",
                metadata={"source": "config_error"}
            )]
        
        try:
            url = "https://openapi.naver.com/v1/search/blog.json"
            headers = {
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret
            }
            params = {
                "query": f"{query} 레시피 재료",
                "display": min(max_results * 2, 10),
                "sort": "sim"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                items = response.json().get("items", [])
                print(f"   검색 성공: {len(items)}개")
                
                return self._format_results(items[:max_results])
            
            elif response.status_code == 429:
                print("   API 호출 제한 초과")
                return [Document(
                    page_content="API 호출 제한을 초과했습니다.",
                    metadata={"source": "rate_limit"}
                )]
            
            else:
                print(f"   API 에러: {response.status_code}")
                return [Document(
                    page_content=f"검색 중 오류: {response.status_code}",
                    metadata={"source": "api_error"}
                )]
        
        except Exception as e:
            print(f"   검색 실패: {e}")
            return [Document(
                page_content=f"검색 중 오류: {str(e)}",
                metadata={"source": "exception"}
            )]
    
    def _format_results(self, items: List[Dict]) -> List[Document]:
        """검색 결과 포맷팅"""
        import re
        
        def clean_html(text):
            text = re.sub('<[^<]+?>', '', text)
            text = text.replace('&quot;', '"')
            text = text.replace('&apos;', "'")
            text = text.replace('&amp;', '&')
            return text
        
        documents = []
        for i, item in enumerate(items, 1):
            title = clean_html(item.get('title', ''))
            description = clean_html(item.get('description', ''))
            link = item.get('link', '')
            
            content = f"[검색 결과 {i}]\n제목: {title}\n\n상세 내용:\n{description}\n\n링크: {link}"
            
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "naver_blog",
                    "title": title,
                    "link": link
                }
            ))
        
        return documents


class GoogleCustomSearch(WebSearchService):
    """구글 커스텀 검색"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """구글 커스텀 검색"""
        print(f"[GoogleSearch] 검색 쿼리: {query}")
        
        if not self.api_key or not self.search_engine_id:
            print("   구글 API 키 없음")
            return [Document(
                page_content="구글 API 키가 필요합니다.",
                metadata={"source": "config_error"}
            )]
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": f"{query} 레시피 만드는법",
                "num": min(max_results, 10),
                "lr": "lang_ko"  # 한국어 결과
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                items = response.json().get("items", [])
                print(f"   검색 성공: {len(items)}개")
                
                return self._format_results(items)
            
            else:
                print(f"   API 에러: {response.status_code}")
                return [Document(
                    page_content=f"검색 중 오류: {response.status_code}",
                    metadata={"source": "api_error"}
                )]
        
        except Exception as e:
            print(f"   검색 실패: {e}")
            return [Document(
                page_content=f"검색 중 오류: {str(e)}",
                metadata={"source": "exception"}
            )]
    
    def _format_results(self, items: List[Dict]) -> List[Document]:
        """검색 결과 포맷팅"""
        documents = []
        
        for i, item in enumerate(items, 1):
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            
            content = f"[검색 결과 {i}]\n제목: {title}\n\n내용:\n{snippet}\n\n링크: {link}"
            
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "google_search",
                    "title": title,
                    "link": link
                }
            ))
        
        return documents


class SerperDevSearch(WebSearchService):
    """Serper.dev 웹 검색"""

    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")

    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """Serper.dev 검색"""
        print(f"[SerperSearch] 검색 쿼리: {query}")

        if not self.api_key:
            print("   Serper API 키 없음")
            return [Document(
                page_content="Serper API 키가 필요합니다.",
                metadata={"source": "config_error"}
            )]

        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": f"{query} 레시피 재료",
                "gl": "kr",
                "hl": "ko",
                "num": max_results
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                organic = data.get("organic", [])
                print(f"   검색 성공: {len(organic)}개")
                return self._format_results(organic[:max_results])

            elif response.status_code == 429:
                print("   API 호출 제한 초과")
                return [Document(
                    page_content="API 호출 제한을 초과했습니다.",
                    metadata={"source": "rate_limit"}
                )]

            else:
                print(f"   API 에러: {response.status_code}")
                return [Document(
                    page_content=f"검색 중 오류: {response.status_code}",
                    metadata={"source": "api_error"}
                )]

        except Exception as e:
            print(f"   검색 실패: {e}")
            return [Document(
                page_content=f"검색 중 오류: {str(e)}",
                metadata={"source": "exception"}
            )]

    def _format_results(self, items: List[Dict]) -> List[Document]:
        """검색 결과 포맷팅"""
        documents = []

        for i, item in enumerate(items, 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")

            content = f"[검색 결과 {i}]\n제목: {title}\n\n내용:\n{snippet}\n\n링크: {link}"

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": "serper_dev",
                    "title": title,
                    "link": link
                }
            ))

        return documents


# 검색 엔진 팩토리
def get_search_service(engine: str = "serper") -> WebSearchService:
    """검색 엔진 선택"""
    engines = {
        "naver": NaverBlogSearch,
        "google": GoogleCustomSearch,
        "serper": SerperDevSearch
    }

    engine_class = engines.get(engine.lower())
    if not engine_class:
        print(f"   지원하지 않는 검색 엔진: {engine}")
        return SerperDevSearch()  # 기본값

    return engine_class()