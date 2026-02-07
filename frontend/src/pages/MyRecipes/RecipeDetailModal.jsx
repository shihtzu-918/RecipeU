// src/pages/MyRecipes/RecipeDetailModal.jsx
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { RECIPE_IMAGES } from "@/images";
import "./RecipeDetailModal.css";

const API_URL = import.meta.env.VITE_API_URL || "";

export default function RecipeDetailModal({ recipe, onClose, onDelete }) {
  const navigate = useNavigate();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // API 응답 구조에 따라 데이터 추출
  const recipeData = (() => {
    if (recipe.recipe && typeof recipe.recipe === "object") {
      return recipe.recipe;
    }
    if (typeof recipe.recipe_json === "string") {
      return JSON.parse(recipe.recipe_json);
    }
    if (recipe.recipe_json && typeof recipe.recipe_json === "object") {
      return recipe.recipe_json;
    }
    return recipe;
  })();

  const title = recipeData.title || recipe.title || "";
  const cookTime = recipeData.cook_time || "";
  const level = recipeData.level || "";
  const ingredients = recipeData.ingredients || [];
  const steps = recipeData.steps || [];
  const imageUrl =
    recipe.image || recipeData.image || recipeData.img_url || null;

  // 날짜 포맷
  const createdAt = recipe.created_at
    ? (() => {
        const d = new Date(recipe.created_at);
        const yy = String(d.getFullYear()).slice(2);
        const mm = String(d.getMonth() + 1).padStart(2, "0");
        const dd = String(d.getDate()).padStart(2, "0");
        return `${yy}.${mm}.${dd}`;
      })()
    : "";

  const prevTime = recipe.cooking_time || recipeData.cooking_time || "";

  // 재료 2컬럼
  const midPoint = Math.ceil(ingredients.length / 2);
  const leftColumn = ingredients.slice(0, midPoint);
  const rightColumn = ingredients.slice(midPoint);

  // 삭제 처리
  const handleDelete = async () => {
    if (isDeleting) return;
    setIsDeleting(true);
    try {
      if (onDelete) {
        await onDelete(recipe.id);
      }
      onClose();
    } catch (err) {
      console.error("삭제 실패:", err);
      alert("삭제에 실패했습니다. 다시 시도해주세요.");
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleStartCook = async () => {
    console.log("[RecipeDetailModal] 요리 시작");
    console.log("원본 recipe:", recipe);
    console.log("recipeData:", recipeData);

    const newSessionId = crypto.randomUUID();
    console.log("[RecipeDetailModal] 세션 ID 생성:", newSessionId);

    // 로그인된 사용자의 member_id 가져오기
    const memberStr = localStorage.getItem("member");
    const member = memberStr ? JSON.parse(memberStr) : null;
    const memberId = member?.id || 0;

    let dbSessionId = null;

    // member_id가 있으면 DB 세션 생성
    if (memberId > 0) {
      try {
        const res = await fetch(`${API_URL}/api/voice/session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ member_id: memberId }),
        });
        if (res.ok) {
          const data = await res.json();
          dbSessionId = data.session_id;
          console.log("[RecipeDetailModal] DB 세션 생성:", dbSessionId);
        }
      } catch (err) {
        console.error("[RecipeDetailModal] DB 세션 생성 실패:", err);
      }
    }

    // 기본 레시피 데이터
    let finalRecipeData = {
      title: title,
      intro: recipeData.intro || "",
      cook_time: cookTime,
      level: level,
      servings: recipeData.servings || recipeData.portion || "2인분",
      ingredients: ingredients,
      steps: steps,
      image: imageUrl,
    };

    // API에서 최신 레시피 데이터 가져오기 시도
    try {
      let url = "";

      // 마이 레시피 router 이용
      if (recipe.id) {
        console.log("마이 레시피 ID:", recipe.id);
        url = `${API_URL}/api/recipe/${recipe.id}`;
      }
      // 랭킹 레시피 router 이용
      else if (recipe.recipe_id) {
        console.log("랭킹 레시피 ID:", recipe.recipe_id);
        url = `${API_URL}/api/rankings/recipes/${recipe.recipe_id}`;
      }

      if (url) {
        console.log("요청 URL:", url);
        const res = await fetch(url);

        if (res.ok) {
          const data = await res.json();
          console.log("API 응답 데이터:", data);

          const apiRecipeData = data.recipe || data;

          // API 응답으로 업데이트
          finalRecipeData = {
            title: apiRecipeData.title || title,
            intro: apiRecipeData.intro || "",
            cook_time: apiRecipeData.cook_time || cookTime,
            level: apiRecipeData.level || level,
            servings:
              apiRecipeData.servings || apiRecipeData.portion || "2인분",
            ingredients: apiRecipeData.ingredients || ingredients,
            steps: apiRecipeData.steps || steps,
            image: apiRecipeData.image || apiRecipeData.img_url || imageUrl,
          };

          console.log("최종 레시피 데이터 (API 응답):", finalRecipeData);
        } else {
          console.log("API 호출 실패, 기본 데이터 사용");
        }
      }
    } catch (err) {
      console.error("API 호출 오류, 기본 데이터 사용:", err);
    }

    navigate({
      to: "/recipe-result",
      state: {
        recipe: finalRecipeData,
        imageUrl: finalRecipeData.image,
        userId: memberId || null,
        title: finalRecipeData.title,
        constraints: null,
        sessionId: newSessionId,
        dbSessionId: dbSessionId,
        generateId: null,
        memberInfo: {
          names: ["나"],
          member_id: memberId,
          allergies: [],
          dislikes: [],
          cooking_tools: [],
        },
        chatHistory: [],
        remainingCount: 1,
        fromMyPage: true,
      },
    });
  };

  return (
    <div className="detail-overlay" onClick={onClose}>
      <div className="detail-page-wrap" onClick={(e) => e.stopPropagation()}>
        {/* 오렌지 클립 */}
        <div className="detail-clip">
          <img src={RECIPE_IMAGES["my-recipe-clip-orange"]} alt="clip" />
        </div>

        {/* 모달 본체 */}
        <div
          className="detail-modal"
          style={{
            backgroundImage: `url(${RECIPE_IMAGES["my-recipe-borderline-beige"]})`,
          }}
        >
          {/* 닫기 버튼 */}
          <button className="detail-close" onClick={onClose}>
            <img src={RECIPE_IMAGES["my-recipe-close"]} alt="close" />
          </button>

          {/* 삭제 버튼 */}
          {onDelete && (
            <button
              className="detail-delete"
              onClick={() => setShowDeleteConfirm(true)}
              title="레시피 삭제"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6" />
              </svg>
            </button>
          )}

          {/* 고정 영역: 날짜, 제목, 이미지, 메타 */}
          <div className="detail-header">
            {/* 날짜 */}
            {createdAt && (
              <div style={{ display: "flex", justifyContent: "center" }}>
                <p className="detail-date">{createdAt}</p>
              </div>
            )}

            {/* 제목 + 밑줄 */}
            <h2 className="detail-titles">{title}</h2>

            {/* 이전 소요시간 */}
            <p className="detail-prev-time">
              이전 소요시간 {prevTime || "00:00:00"}
            </p>

            {/* 이미지 */}
            <div className="detail-image-wrap">
              {imageUrl ? (
                <img src={imageUrl} alt={title} className="detail-image" />
              ) : (
                <div className="detail-image-placeholder">
                  <svg width="60" height="60" viewBox="0 0 60 60" fill="none">
                    <circle
                      cx="30"
                      cy="30"
                      r="27"
                      stroke="#C4956A"
                      strokeWidth="2"
                      fill="none"
                    />
                    <path
                      d="M18 38C18 38 24 27 30 27C36 27 42 38 42 38"
                      stroke="#C4956A"
                      strokeWidth="2"
                    />
                    <circle cx="23" cy="24" r="3" fill="#C4956A" />
                  </svg>
                </div>
              )}
            </div>

            {/* 시간 & 난이도 */}
            <div className="detail-meta">
              {cookTime && (
                <span className="meta-item">
                  <img
                    src={RECIPE_IMAGES["my-recipe-time"]}
                    alt="time"
                    className="meta-icon"
                  />
                  {cookTime}
                </span>
              )}
              {level && (
                <span className="meta-item">
                  <img
                    src={RECIPE_IMAGES["my-recipe-level"]}
                    alt="level"
                    className="meta-icon"
                  />
                  {level}
                </span>
              )}
            </div>
          </div>

          {/* 스크롤 영역: 재료 + 조리법 */}
          <div className="detail-content">
            {/* 재료 */}
            <div className="detail-section">
              <h3 className="detail-section-title">재료</h3>
              <hr className="detail-section-line" />
              <div className="detail-ingredients">
                {ingredients.length > 0 ? (
                  <>
                    <div className="ingredients-columns">
                      {leftColumn.map((ing, idx) => (
                        <div key={idx} className="ingredient-item">
                          <span>• </span>
                          <span>
                            {ing.name} {ing.amount}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div className="ingredients-columns">
                      {rightColumn.map((ing, idx) => (
                        <div key={idx} className="ingredient-item">
                          <span>• </span>
                          <span>
                            {ing.name} {ing.amount}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="detail-empty-text">재료 정보가 없습니다</p>
                )}
              </div>
            </div>

            {/* 조리법 */}
            <div className="detail-section">
              <h3 className="detail-section-title">조리법</h3>
              <hr className="detail-section-line" />
              <ol className="detail-steps">
                {steps.length > 0 ? (
                  steps.map((step, idx) => (
                    <li key={idx} className="step-item">
                      {step.desc || step}
                    </li>
                  ))
                ) : (
                  <p className="detail-empty-text">조리법 정보가 없습니다</p>
                )}
              </ol>
            </div>

            {/* 하단 요리 시작 버튼 */}
            <div className="detail-action-area">
              <button className="start-cook-btn" onClick={handleStartCook}>
                요리 시작하기
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 삭제 확인 모달 */}
      {showDeleteConfirm && (
        <div
          className="delete-confirm-overlay"
          onClick={() => setShowDeleteConfirm(false)}
        >
          <div
            className="delete-confirm-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="delete-confirm-text">이 레시피를 삭제하시겠습니까?</p>
            <p className="delete-confirm-subtext">
              삭제된 레시피는 복구할 수 없습니다.
            </p>
            <div className="delete-confirm-buttons">
              <button
                className="delete-confirm-btn cancel"
                onClick={() => setShowDeleteConfirm(false)}
                disabled={isDeleting}
              >
                취소
              </button>
              <button
                className="delete-confirm-btn confirm"
                onClick={handleDelete}
                disabled={isDeleting}
              >
                {isDeleting ? "삭제 중..." : "삭제"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
