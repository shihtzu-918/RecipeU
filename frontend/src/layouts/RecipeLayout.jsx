// src/layouts/RecipeLayout.jsx
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import RecipeBottomSheet from "@/components/RecipeBottomSheet";
import { RECIPE_IMAGES } from "@/images";
import "./RecipeLayout.css";

export default function RecipeLayout({
  children,
  steps = [],
  currentStep = 1,
  showBottomSheet = true,
  onStepClick = null,
  backgroundColor = "brown", // 배경색 선택 (brown, green, yellow)
}) {
  const navigate = useNavigate();
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  // 배경 이미지 매핑
  const bgImages = {
    brown: RECIPE_IMAGES["cook-bg-brown"],
    green: RECIPE_IMAGES["cook-bg-green"],
    yellow: RECIPE_IMAGES["cook-bg-yellow"],
  };

  return (
    <div
      className="recipe-layout-container"
      style={{ backgroundImage: `url(${bgImages[backgroundColor]})` }}
    >
      {/* 오버레이 */}
      <div
        className={`recipe-overlay ${isSheetOpen ? "active" : ""}`}
        onClick={() => setIsSheetOpen(false)}
      />

      {/* 헤더 - 마스코트 + 닫기 버튼 */}
      <div className="recipe-header">
        <img
          src={RECIPE_IMAGES["chef-mascot"]}
          alt="요리사 마스코트"
          className="recipe-mascot"
        />
        <button
          className="recipe-close"
          onClick={() => navigate({ to: "/home" })}
        >
          <img
            src={RECIPE_IMAGES["exit-icon"]}
            alt="닫기"
            className="close-icon"
          />
        </button>
      </div>

      {/* 메인 카드 */}
      <div className="main-recipe-card">
        {/* 스크롤 가능한 콘텐츠 영역 */}
        <div className="main-recipe-card-content">{children}</div>

        {/* 레시피 전체보기 - 옵션 */}
        {showBottomSheet && steps.length > 0 && (
          <RecipeBottomSheet
            steps={steps}
            currentStep={currentStep}
            isOpen={isSheetOpen}
            setIsOpen={setIsSheetOpen}
            onStepClick={onStepClick}
          />
        )}
      </div>
    </div>
  );
}
