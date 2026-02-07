// src/pages/MyRecipes/MyRecipesPage.jsx
import { useState, useEffect } from "react";
import RecipeDetailModal from "./RecipeDetailModal";
import BottomNav from "@/components/BottomNav";
import { RECIPE_IMAGES } from "@/images";
import "./MyRecipesPage.css";

function StarRating({ rating = 0, size = 11 }) {
  const stars = [];
  for (let i = 1; i <= 3; i++) {
    stars.push(
      <span
        key={i}
        className={`star ${i <= rating ? "star-filled" : "star-empty"}`}
        style={{ fontSize: size }}
      >
        ★
      </span>,
    );
  }
  return <div className="card-star-overlay">{stars}</div>;
}

export default function MyRecipesPage() {
  const API_URL = import.meta.env.VITE_API_URL || "";
  const [recipes, setRecipes] = useState([]);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [loading, setLoading] = useState(true);

  // 로그인된 회원 정보
  const memberStr = localStorage.getItem("member");
  const member = memberStr ? JSON.parse(memberStr) : null;
  const memberId = member?.id || 0;

  useEffect(() => {
    fetchMyRecipes();
  }, []);

  const fetchMyRecipes = async () => {
    try {
      const res = await fetch(
        `${API_URL}/api/recipe/list?member_id=${memberId}`,
      );
      if (res.ok) {
        const data = await res.json();
        setRecipes(data.recipes || []);
      }
    } catch (err) {
      console.log("레시피 목록 불러오기 실패:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRecipeClick = async (recipe) => {
    try {
      const res = await fetch(`${API_URL}/api/recipe/${recipe.id}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedRecipe(data);
      }
    } catch (err) {
      setSelectedRecipe(recipe);
    }
  };

  // 레시피 삭제
  const handleDeleteRecipe = async (recipeId) => {
    const res = await fetch(`${API_URL}/api/recipe/${recipeId}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      throw new Error("삭제 실패");
    }
    // 목록에서 제거
    setRecipes(recipes.filter((r) => r.id !== recipeId));
  };

  const isEmpty = !loading && recipes.length === 0;

  return (
    <div
      className="my-recipes-page"
      style={{ backgroundImage: `url(${RECIPE_IMAGES["cook-bg-brown"]})` }}
    >
      {/* 내부 스크롤 영역 */}
      <div className={`my-recipes-scroll ${isEmpty ? "is-empty" : ""}`}>
        {/* 클립 이미지 (베이지) */}
        <div className="clipboard-clip">
          <img
            src={RECIPE_IMAGES["my-recipe-clip-beige"]}
            alt="clip"
            loading="lazy"
          />
        </div>

        {/* 클립보드 본체 */}
        <div className={`clipboard-board ${isEmpty ? "is-empty" : ""}`}>
          <h1 className="clipboard-title">마이레시피</h1>

          {loading && (
            <div className="recipes-loading">
              <p>불러오는 중...</p>
            </div>
          )}

          {isEmpty && (
            <div className="recipes-empty">
              <p className="empty-message">요리를 시작하러 가볼까요?</p>
            </div>
          )}

          {!isEmpty && !loading && (
            <div className="recipes-grid-container">
              <div className="recipes-grid">
                {recipes.map((recipe) => (
                  <div
                    key={recipe.id}
                    className="recipe-cards"
                    onClick={() => handleRecipeClick(recipe)}
                  >
                    <div className="recipe-cards-image">
                      {recipe.image ? (
                        <img
                          src={recipe.image}
                          alt={recipe.title}
                          loading="lazy"
                        />
                      ) : (
                        <div className="recipe-cards-placeholder">
                          <svg
                            width="32"
                            height="32"
                            viewBox="0 0 32 32"
                            fill="none"
                          >
                            <circle
                              cx="16"
                              cy="16"
                              r="14"
                              stroke="#C4956A"
                              strokeWidth="1.5"
                              fill="none"
                            />
                            <path
                              d="M10 20C10 20 13 15 16 15C19 15 22 20 22 20"
                              stroke="#C4956A"
                              strokeWidth="1.5"
                            />
                            <circle cx="12" cy="13" r="1.5" fill="#C4956A" />
                          </svg>
                        </div>
                      )}
                      <StarRating rating={recipe.rating || 3} size={11} />
                    </div>
                    <span className="recipe-cards-title">{recipe.title}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 레시피 상세 모달 */}
      {selectedRecipe && (
        <RecipeDetailModal
          recipe={selectedRecipe}
          onClose={() => setSelectedRecipe(null)}
          onDelete={handleDeleteRecipe}
        />
      )}

      {/* 하단 고정 네비게이션 */}
      <BottomNav />
    </div>
  );
}
