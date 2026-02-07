# backend/features/auth/router.py
"""
네이버 OAuth 로그인 API
"""
import secrets
import httpx
from fastapi import APIRouter, HTTPException, Query
from app.config import settings
from models.mysql_db import upsert_member

router = APIRouter()

# 네이버 OAuth 엔드포인트
NAVER_AUTH_URL = "https://nid.naver.com/oauth2.0/authorize"
NAVER_TOKEN_URL = "https://nid.naver.com/oauth2.0/token"
NAVER_PROFILE_URL = "https://openapi.naver.com/v1/nid/me"


@router.get("/login-url")
async def get_naver_login_url(callback_url: str = Query(...)):
    """
    프론트엔드가 리다이렉트할 네이버 인증 URL 반환.
    callback_url: 프론트엔드의 /naver-callback 절대 URL
    """
    if not settings.NAVER_CLIENT_ID:
        raise HTTPException(status_code=500, detail="NAVER_CLIENT_ID가 설정되지 않았습니다.")

    state = secrets.token_urlsafe(16)

    url = (
        f"{NAVER_AUTH_URL}"
        f"?response_type=code"
        f"&client_id={settings.NAVER_CLIENT_ID}"
        f"&redirect_uri={callback_url}"
        f"&state={state}"
    )
    return {"url": url, "state": state}


@router.post("/callback")
async def naver_callback(code: str = Query(...), state: str = Query(...),
                         callback_url: str = Query(...)):
    """
    프론트엔드가 네이버에서 받은 code/state를 전달하면:
    1) 액세스 토큰 발급
    2) 네이버 프로필 조회
    3) member 테이블 upsert
    4) 회원 정보 반환
    """
    # 1. 액세스 토큰 발급
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            NAVER_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": settings.NAVER_CLIENT_ID,
                "client_secret": settings.NAVER_CLIENT_SECRET,
                "code": code,
                "state": state,
                "redirect_uri": callback_url,
            },
        )

    token_data = token_resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail=f"토큰 발급 실패: {token_data}")

    # 2. 네이버 프로필 조회
    async with httpx.AsyncClient() as client:
        profile_resp = await client.get(
            NAVER_PROFILE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    profile_json = profile_resp.json()
    if profile_json.get("resultcode") != "00":
        raise HTTPException(status_code=400, detail=f"프로필 조회 실패: {profile_json}")

    naver = profile_json["response"]

    # 3. DB upsert용 데이터 매핑 (새 스키마: id, naver_id, email, nickname, birthday, mem_photo, mem_type, to_cnt, first_visit, last_visit, member_del)
    member_data = {
        "naver_id": naver.get("id", ""),
        "email": naver.get("email", ""),
        "nickname": naver.get("nickname", ""),
        "birthday": naver.get("birthday", ""),
        "mem_photo": naver.get("profile_image", ""),
        "mem_type": "NAVER",
    }

    # 4. member 테이블 upsert
    member = upsert_member(member_data)

    # 5. 프론트엔드에 필요한 형식으로 응답 구성
    response_member = {
        "id": member["id"],
        "nickname": member["nickname"],
        "email": member["email"],
        "name": member["nickname"],  # 프론트엔드에서 name 필드도 사용
        "birthday": member.get("birthday", ""),  # 생일 필드 명시적 포함
        "mem_photo": member.get("mem_photo", ""),
        "profile_image": member.get("mem_photo", None),  # 호환성을 위해 추가
    }

    return {"member": response_member}
