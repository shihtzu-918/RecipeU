# backend/features/user/router.py
"""
User REST API 라우터 - MySQL 기반 (CRUD 전체)
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from pydantic import BaseModel

from models.mysql_db import (
    get_member_by_id, get_member_personalization, upsert_member_personalization,
    get_families, get_family_personalization, upsert_family_personalization,
    add_family, update_family, delete_family,
    get_member_utensils, get_all_utensils, set_member_utensils,
    load_mypage_data
)


# Request 스키마
class PersonalizationUpdate(BaseModel):
    allergies: List[str] = []
    dislikes: List[str] = []


class FamilyCreate(BaseModel):
    relationship: str = ""


class FamilyUpdate(BaseModel):
    relationship: str = ""
    allergies: List[str] = []
    dislikes: List[str] = []


class UtensilsUpdate(BaseModel):
    utensil_ids: List[int] = []

router = APIRouter()


@router.get("/profile")
async def get_profile(member_id: int = Query(default=0)):
    """사용자 프로필 조회 (MySQL)"""
    if member_id == 0:
        return {
            "name": "게스트",
            "allergies": [],
            "dislikes": []
        }
    member = get_member_by_id(member_id)
    if not member:
        raise HTTPException(status_code=404, detail="회원을 찾을 수 없습니다")
    psnl = get_member_personalization(member_id)
    return {
        "id": member.get("id"),
        "name": member.get("nickname", "사용자"),
        "email": member.get("email"),
        "birthday": member.get("birthday", ""),
        "allergies": psnl.get("allergies", []) if psnl else [],
        "dislikes": psnl.get("dislikes", []) if psnl else []
    }

@router.get("/family")
async def get_family_info(member_id: int = Query(default=0)):
    """가족 구성원 정보 조회 (MySQL)"""
    if member_id == 0:
        return {"family_members": []}

    families = get_families(member_id)
    result = []

    for f in families:
        psnl = get_family_personalization(f["id"])
        result.append({
            "id": f["id"],
            "relationship": f.get("relationship", ""),
            "allergies": psnl.get("allergies", []) if psnl else [],
            "dislikes": psnl.get("dislikes", []) if psnl else []
        })

    return {"family_members": result}


@router.get("/family/{family_id}")
async def get_family_member_info(family_id: int) -> Dict[str, Any]:
    """특정 가족 구성원 정보 조회 (MySQL)"""
    psnl = get_family_personalization(family_id)

    return {
        "id": family_id,
        "allergies": psnl.get("allergies", []) if psnl else [],
        "dislikes": psnl.get("dislikes", []) if psnl else []
    }


@router.get("/all-constraints")
async def get_all_constraints(member_id: int = Query(default=0)):
    """회원 + 가족 전체의 알레르기/비선호 통합 조회"""
    if member_id == 0:
        return {
            "allergies": [],
            "dislikes": [],
            "utensils": []
        }

    all_allergies = set()
    all_dislikes = set()

    # 본인
    member_psnl = get_member_personalization(member_id)
    if member_psnl:
        all_allergies.update(member_psnl.get("allergies", []))
        all_dislikes.update(member_psnl.get("dislikes", []))

    # 가족
    families = get_families(member_id)
    for f in families:
        psnl = get_family_personalization(f["id"])
        if psnl:
            all_allergies.update(psnl.get("allergies", []))
            all_dislikes.update(psnl.get("dislikes", []))

    # 조리도구
    utensil_ids = get_member_utensils(member_id)
    all_utensils = get_all_utensils()
    member_utensils = [u["name"] for u in all_utensils if u["id"] in utensil_ids]

    return {
        "allergies": list(all_allergies),
        "dislikes": list(all_dislikes),
        "utensils": member_utensils
    }


# ══════════════════════════════════════════════════════════════
# 마이페이지 전체 로드 (한 번에 모든 데이터)
# ══════════════════════════════════════════════════════════════

@router.get("/mypage")
async def get_mypage_data(member_id: int = Query(default=0)):
    """마이페이지 전체 데이터 조회 (회원 개인화 + 가족 + 조리도구)"""
    if member_id == 0:
        return {
            "personalization": {"allergies": [], "dislikes": []},
            "families": [],
            "utensils": [],
            "member_utensil_ids": []
        }

    try:
        return load_mypage_data(member_id)
    except Exception as e:
        print(f"[User API] 마이페이지 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# 회원 개인화 (allergies, dislikes) CRUD
# ══════════════════════════════════════════════════════════════

@router.put("/personalization")
async def update_personalization(
    data: PersonalizationUpdate,
    member_id: int = Query(...)
):
    """회원 본인 개인화 수정"""
    if member_id == 0:
        raise HTTPException(status_code=400, detail="로그인이 필요합니다")

    try:
        result = upsert_member_personalization(
            member_id=member_id,
            allergies=data.allergies,
            dislikes=data.dislikes
        )
        return {
            "success": True,
            "personalization": {
                "allergies": result.get("allergies", []),
                "dislikes": result.get("dislikes", [])
            }
        }
    except Exception as e:
        print(f"[User API] 개인화 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# 가족 CRUD
# ══════════════════════════════════════════════════════════════

@router.post("/family")
async def create_family(
    data: FamilyCreate,
    member_id: int = Query(...)
):
    """가족 구성원 추가"""
    if member_id == 0:
        raise HTTPException(status_code=400, detail="로그인이 필요합니다")

    try:
        result = add_family(member_id=member_id, relationship=data.relationship)
        return {
            "success": True,
            "family": {
                "id": result.get("id"),
                "relationship": result.get("relationship", "")
            }
        }
    except Exception as e:
        print(f"[User API] 가족 추가 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/family/{family_id}")
async def update_family_member(
    family_id: int,
    data: FamilyUpdate,
    member_id: int = Query(...)
):
    """가족 구성원 수정 (관계 + 개인화)"""
    if member_id == 0:
        raise HTTPException(status_code=400, detail="로그인이 필요합니다")

    try:
        # 관계 수정
        update_family(family_id=family_id, relationship=data.relationship)

        # 개인화 수정
        psnl = upsert_family_personalization(
            member_id=member_id,
            family_id=family_id,
            allergies=data.allergies,
            dislikes=data.dislikes
        )

        return {
            "success": True,
            "family": {
                "id": family_id,
                "relationship": data.relationship,
                "allergies": psnl.get("allergies", []),
                "dislikes": psnl.get("dislikes", [])
            }
        }
    except Exception as e:
        print(f"[User API] 가족 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/family/{family_id}")
async def delete_family_member(family_id: int, member_id: int = Query(default=0)):
    """가족 구성원 삭제"""
    if member_id == 0:
        raise HTTPException(status_code=400, detail="로그인이 필요합니다")

    try:
        delete_family(family_id=family_id)
        return {"success": True, "message": "가족 구성원이 삭제되었습니다"}
    except Exception as e:
        print(f"[User API] 가족 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# 조리도구 CRUD
# ══════════════════════════════════════════════════════════════

@router.get("/utensils")
async def get_utensils():
    """전체 조리도구 목록 조회"""
    try:
        return {"utensils": get_all_utensils()}
    except Exception as e:
        print(f"[User API] 조리도구 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/utensils")
async def update_member_utensils(
    data: UtensilsUpdate,
    member_id: int = Query(...)
):
    """회원 조리도구 전체 교체"""
    if member_id == 0:
        raise HTTPException(status_code=400, detail="로그인이 필요합니다")

    try:
        set_member_utensils(member_id=member_id, utensil_ids=data.utensil_ids)
        return {
            "success": True,
            "utensil_ids": data.utensil_ids
        }
    except Exception as e:
        print(f"[User API] 조리도구 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    