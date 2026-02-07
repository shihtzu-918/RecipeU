# backend/features/mypage/router.py
"""
마이페이지 REST API
- 통합 조회 (게스트 분기 포함)
- 가족 CRUD
- 개인화(알레르기/비선호) CRUD
- 조리도구 CRUD
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from models.mysql_db import (
    load_mypage_data,
    add_family,
    update_family,
    delete_family,
    upsert_member_personalization,
    upsert_family_personalization,
    get_all_utensils,
    get_member_utensils,
    set_member_utensils,
    seed_utensils,
)

router = APIRouter()

# 게스트 계정 ID (개인화/마이레시피 저장 불가)
GUEST_MEMBER_ID = 2


# ── 요청 스키마 ──

class FamilyCreate(BaseModel):
    relationship: str = ""


class FamilyUpdate(BaseModel):
    relationship: str


class PersonalizationUpdate(BaseModel):
    allergies: List[str] = []
    dislikes: List[str] = []


class UtensilUpdate(BaseModel):
    utensil_ids: List[int] = []


# ── 조리도구 시딩 ──

TOOL_LIST = [
    "밥솥", "전자레인지", "오븐", "에어프라이어", "찜기",
    "믹서기", "착즙기", "커피머신", "토스트기", "와플메이커"
]


def init_utensils():
    """서버 시작 시 조리도구 마스터 데이터 시딩"""
    try:
        seed_utensils(TOOL_LIST)
        print("조리도구 마스터 데이터 시딩 완료")
    except Exception as e:
        print(f"조리도구 시딩 실패 (MySQL 미연결?): {e}")


# ── API 엔드포인트 ──

@router.get("/guest")
async def get_guest_defaults():
    """
    게스트 모드용 기본 데이터
    - 로그인 안 한 사용자를 위한 조리도구 목록만 반환
    - 프론트엔드에서 localStorage로 관리
    """
    try:
        utensils = get_all_utensils()
        return {
            "is_guest": True,
            "personalization": {"allergies": [], "dislikes": []},
            "families": [],
            "utensils": utensils,
            "member_utensil_ids": [],
        }
    except Exception as e:
        # DB 연결 안 되어도 기본값 반환
        return {
            "is_guest": True,
            "personalization": {"allergies": [], "dislikes": []},
            "families": [],
            "utensils": [],
            "member_utensil_ids": [],
        }


@router.get("/{member_id}")
async def get_mypage(member_id: int):
    """
    마이페이지 전체 데이터 조회
    - member_id가 0 또는 게스트(2)이면 게스트 모드 응답
    """
    if member_id in [0, GUEST_MEMBER_ID]:
        return await get_guest_defaults()

    try:
        data = load_mypage_data(member_id)
        data["is_guest"] = False
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{member_id}/family")
async def create_family(member_id: int, body: FamilyCreate):
    """가족 추가"""
    if member_id in [0, GUEST_MEMBER_ID]:
        raise HTTPException(status_code=400, detail="게스트는 가족을 추가할 수 없습니다. 로그인해주세요.")

    try:
        family = add_family(member_id, body.relationship)
        return family
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/family/{family_id}")
async def modify_family(family_id: int, body: FamilyUpdate):
    """가족 관계 수정"""
    try:
        family = update_family(family_id, body.relationship)
        return family
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/family/{family_id}")
async def remove_family(family_id: int):
    """가족 삭제"""
    try:
        delete_family(family_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{member_id}/personalization")
async def update_member_personalization(member_id: int, body: PersonalizationUpdate):
    """회원 본인 알레르기/비선호 수정"""
    if member_id in [0, GUEST_MEMBER_ID]:
        raise HTTPException(status_code=400, detail="게스트는 서버에 저장할 수 없습니다. 로그인해주세요.")

    try:
        result = upsert_member_personalization(member_id, body.allergies, body.dislikes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{member_id}/family/{family_id}/personalization")
async def update_family_personalization(member_id: int, family_id: int, body: PersonalizationUpdate):
    """가족 알레르기/비선호 수정"""
    if member_id in [0, GUEST_MEMBER_ID]:
        raise HTTPException(status_code=400, detail="게스트는 서버에 저장할 수 없습니다. 로그인해주세요.")

    try:
        result = upsert_family_personalization(member_id, family_id, body.allergies, body.dislikes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{member_id}/utensils")
async def get_utensils(member_id: int):
    """조리도구 전체 목록 + 회원 보유 목록"""
    try:
        all_utensils = get_all_utensils()
        member_ids = get_member_utensils(member_id) if member_id > 0 else []
        return {"utensils": all_utensils, "member_utensil_ids": member_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{member_id}/utensils")
async def update_utensils(member_id: int, body: UtensilUpdate):
    """회원 조리도구 갱신"""
    if member_id in [0, GUEST_MEMBER_ID]:
        raise HTTPException(status_code=400, detail="게스트는 서버에 저장할 수 없습니다. 로그인해주세요.")

    try:
        set_member_utensils(member_id, body.utensil_ids)
        return {"ok": True, "utensil_ids": body.utensil_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
