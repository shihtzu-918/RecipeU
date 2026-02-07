// src/pages/Recipe/RecipeResultPage.jsx
import { useNavigate, useLocation } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import RecipeLayout from "@/layouts/RecipeLayout";
import ButtonRed from "@/components/ButtonRed";
import ButtonWhite from "@/components/ButtonWhite";
import { RECIPE_IMAGES } from "@/images";
import "./RecipeResultPage.css";

export default function RecipeResultPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // location.state에서 가져오기
  const {
    recipe,
    userId,
    title,
    constraints,
    sessionId,
    dbSessionId,
    generateId,
    memberInfo,
    chatHistory,
    remainingCount: initialCount,
    imageUrl,
    fromMyPage,
  } = location.state || {};

  const [remainingCount, setRemainingCount] = useState(
    initialCount !== undefined ? initialCount : 1,
  );

  const [isFlipped, setIsFlipped] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL || "";

  // useEffect 수정 - 한 번만 실행
  useEffect(() => {
    if (!recipe) {
      console.warn("[RecipeResultPage] 레시피 데이터 없음 - 대기 중...");

      // 약간의 딜레이 후 다시 확인
      const timer = setTimeout(() => {
        if (!recipe && !location.state?.recipe) {
          console.error("[RecipeResultPage] 레시피 데이터 최종 없음");
          navigate({ to: "/home", replace: true });
        }
      }, 100);

      return () => clearTimeout(timer);
    } else {
      console.log("[RecipeResultPage] 받은 레시피:", recipe);
      console.log("[RecipeResultPage] 이미지:", recipe.image || recipe.img_url);
      console.log("[RecipeResultPage] 세션 ID:", sessionId);
      console.log("[RecipeResultPage] 남은 횟수:", remainingCount);
    }
  }, [recipe, navigate, sessionId, remainingCount, location.state]);

  // 초기 로딩 중에는 null 반환하지 않기
  if (!recipe) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
        }}
      >
        <p>레시피를 불러오는 중...</p>
      </div>
    );
  }

  const handleRegenerate = async () => {
    console.log("[RecipeResult] ===== 재생성 시작 =====");
    console.log("[RecipeResult] remainingCount:", remainingCount);

    if (remainingCount <= 0) {
      console.log("[RecipeResult] 남은 횟수 없음!");
      return;
    }

    console.log("[RecipeResult] sessionId:", sessionId);

    if (!sessionId) {
      alert("세션 정보가 없습니다.");
      return;
    }

    // 이미 location.state로 chatHistory, memberInfo를 받았으므로 API 호출 불필요
    console.log("[RecipeResult] ChatPage로 이동 (fromMyPage:", fromMyPage, ")");

    navigate({
      to: "/chat",
      state: {
        sessionId: sessionId,
        existingMessages: chatHistory || [],
        memberInfo: memberInfo,
        recipe: recipe,
        skipToChat: true,
        fromRegenerate: true,
        fromMyPage: fromMyPage || false,
      },
    });
  };

  const handleStartCooking = () => {
    console.log("[RecipeResult] 요리 시작하기 클릭");
    console.log("[RecipeResult] recipe:", recipe);
    console.log("[RecipeResult] imageUrl:", imageUrl);

    const recipeImage =
      imageUrl ||
      recipe?.image ||
      recipe?.img_url ||
      "https://kr.object.ncloudstorage.com/recipu-bucket/assets/default_img.webp";

    console.log("[RecipeResult] 최종 이미지:", recipeImage);

    // steps 정규화: 문자열 배열을 객체 배열로 변환
    const normalizedSteps = (recipe.steps || []).map((step, index) => {
      if (typeof step === "string") {
        return { desc: step, order: index + 1 };
      }
      return step;
    });

    const cookState = {
      recipe: {
        name: recipe.title,
        intro: recipe.intro || "",
        time: recipe.cook_time || "30분",
        level: recipe.level || "중급",
        servings: recipe.servings || recipe.portion || "2인분",
        ingredients: recipe.ingredients || [],
        steps: normalizedSteps,
        image: recipeImage,
      },
      currentStepIndex: 0,
      elapsedTime: 0,
      dbSessionId: dbSessionId || null,
      generateId: generateId || null,
      sessionId: sessionId || null,
      userId: userId || null,
      memberInfo: memberInfo || null,
      chatHistory: chatHistory || null,
      remainingCount: remainingCount,
      imageUrl: imageUrl || recipeImage,
      fromMyPage: fromMyPage || false,
    };

    console.log("[RecipeResult] cookState 저장:", cookState);

    localStorage.setItem("cookState", JSON.stringify(cookState));
    navigate({ to: "/cook" });
  };

  const handleFlipCard = () => {
    setIsFlipped(!isFlipped);
  };

  const recipeImage =
    imageUrl ||
    recipe?.image ||
    recipe?.img_url ||
    "https://kr.object.ncloudstorage.com/recipu-bucket/assets/default_img.webp";

  // steps 정규화: 문자열을 그대로 표시 또는 객체에서 추출
  const normalizedSteps = (recipe.steps || []).map((step) => {
    if (typeof step === "string") {
      return step;
    }
    return step.desc || step.content || step;
  });

  return (
    <RecipeLayout steps={normalizedSteps} currentStep={0}>
      <div className="result-title-section">
        <p className="result-subtitle">오늘의 추천 레시피는</p>
        <h1 className="result-title">
          <span className="highlight">{recipe.title}</span>{" "}
          <span className="result-subtitle">입니다</span>
        </h1>
      </div>

      <div
        className={`result-card-container ${isFlipped ? "flipped" : ""}`}
        onClick={handleFlipCard}
      >
        <div className="result-card">
          <div className="result-card-front">
            <div className="result-image-wrapper">
              <img
                className="result-image"
                src={recipeImage}
                alt={recipe.title}
                onError={(e) => {
                  console.error(
                    "[RecipeResultPage] 이미지 로드 실패:",
                    e.target.src,
                  );
                  e.target.src =
                    "https://kr.object.ncloudstorage.com/recipu-bucket/assets/default_img.webp";
                }}
              />

              <div className="result-image-info">
                <div className="info-badge">
                  <img
                    src={RECIPE_IMAGES["time-icon"]}
                    alt="시간"
                    className="badge-icon"
                  />
                  <span>{recipe.cook_time || "30분"}</span>
                </div>
                <div className="info-badge">
                  <img
                    src={RECIPE_IMAGES["level-icon"]}
                    alt="난이도"
                    className="badge-icon"
                  />
                  <span>{recipe.level || "중급"}</span>
                </div>
              </div>

              <div className="flip-hint">
                <span>재료 보기</span>
                <span className="flip-icon">↻</span>
              </div>
            </div>
          </div>

          <div className="result-card-back">
            <div className="ingredients-wrapper">
              <h3 className="ingredients-title">필요한 재료</h3>
              <div className="ingredients-list">
                {recipe.ingredients && recipe.ingredients.length > 0 ? (
                  recipe.ingredients.map((ingredient, idx) => (
                    <div key={idx} className="ingredient-items">
                      <span className="ingredient-name">
                        {ingredient.name || ingredient}
                      </span>
                      <span className="ingredient-amount">
                        {ingredient.amount || ""}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="no-ingredients">재료 정보가 없습니다</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="result-actions">
        <div className="result-button-wrapper">
          <ButtonRed
            onClick={() => {
              console.log("[RecipeResult] 버튼 클릭됨!");
              handleRegenerate();
            }}
            disabled={remainingCount === 0}
            subText={
              remainingCount > 0 ? `${remainingCount}회 남음` : "재생성 불가"
            }
          >
            다시 생성
          </ButtonRed>
        </div>
        <div className="result-button-wrapper">
          <ButtonWhite onClick={handleStartCooking}>요리 시작하기</ButtonWhite>
        </div>
      </div>
    </RecipeLayout>
  );
}
