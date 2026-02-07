// src/pages/Cook/CookModePage.jsx
"use client";

import { useNavigate } from "@tanstack/react-router";
import { useState, useEffect, useRef } from "react";
import RecipeLayout from "@/layouts/RecipeLayout";
import ButtonRed from "@/components/ButtonRed";
import { RECIPE_IMAGES } from "@/images";
import "./CookModePage.css";

export default function CookModePage() {
  const navigate = useNavigate();
  const hasLoadedRef = useRef(false);

  // localStorage에서 state 복원
  const [cookState] = useState(() => {
    const saved = localStorage.getItem("cookState");
    console.log("[CookMode] localStorage 원본:", saved);

    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log("[CookMode] 파싱된 데이터:", parsed);
        return parsed;
      } catch (err) {
        console.error("[CookMode] JSON 파싱 실패:", err);
      }
    }

    return {
      currentStepIndex: 0,
      elapsedTime: 0,
      recipe: {
        name: "레시피 없음",
        intro: "",
        time: "0분",
        level: "초급",
        servings: "1인분",
        ingredients: [],
        steps: [{ no: 1, desc: "레시피 정보가 없습니다." }],
        image: RECIPE_IMAGES["default_img"],
      },
      cookingFinished: false,
    };
  });

  // cookState에서 값 추출
  const passedStepIndex = cookState.currentStepIndex ?? 0;
  const passedElapsedTime = cookState.elapsedTime ?? 0;
  const voiceSessionId = cookState.voiceSessionId ?? null;
  const memberId =
    cookState.memberId ??
    (() => {
      const m = localStorage.getItem("member");
      return m ? JSON.parse(m).id || 2 : 2;
    })();

  const recipe = cookState.recipe;
  const dbSessionId = cookState.dbSessionId ?? null;
  const generateId = cookState.generateId ?? null;
  const recipeSteps = recipe.steps || [];
  const handleBackToResult = () => {
    navigate({
      to: "/recipe-result",
      state: {
        recipe: {
          title: recipe.name,
          intro: recipe.intro,
          cook_time: recipe.time,
          level: recipe.level,
          servings: recipe.servings,
          ingredients: recipe.ingredients,
          steps: recipe.steps,
          image: recipe.image,
        },
        imageUrl: recipe.image,
        dbSessionId,
        generateId,
        remainingCount: 0, // 쿡모드에서는 재생성 불가
        fromMyPage: true, // 쿡모드에서 온 경우
      },
    });
  };

  const [currentStepIndex, setCurrentStepIndex] = useState(passedStepIndex);
  const [elapsedTime, setElapsedTime] = useState(passedElapsedTime);

  console.log("[CookMode] 레시피:", recipe);
  console.log("[CookMode] 조리 단계:", recipeSteps);

  useEffect(() => {
    return () => {
      const nextPath = window.location.pathname;
      if (nextPath === "/cook-complete") {
        console.log("[CookMode] cook-complete로 이동 - localStorage 삭제");
        localStorage.removeItem("cookState");
      } else {
        console.log("[CookMode] 다른 페이지로 이동 - localStorage 유지");
      }
    };
  }, []);

  const peuImages = [
    "/peu_banjuk.png",
    "/peu_chicken.png",
    "/peu_cook.png",
    "/peu_gimbab.png",
    "/peu_hurai.png",
    "/peu_icecream.png",
    "/peu_ramen.png",
    "/peu_pizza.png",
    "/peu_salad.png",
    "/peu_wink.png",
  ];

  const getRandomPeuImage = (exclude) => {
    if (peuImages.length === 0) return RECIPE_IMAGES["default_img"];
    if (peuImages.length === 1) return peuImages[0];
    let next = peuImages[Math.floor(Math.random() * peuImages.length)];
    while (next === exclude) {
      next = peuImages[Math.floor(Math.random() * peuImages.length)];
    }
    return next;
  };

  const [randomPeuImage, setRandomPeuImage] = useState(() =>
    getRandomPeuImage(),
  );

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  const [slideDir, setSlideDir] = useState("");
  const [isAnimating, setIsAnimating] = useState(false);

  const changeStep = (direction) => {
    if (isAnimating) return;
    const next =
      direction === "next"
        ? Math.min(currentStepIndex + 1, recipeSteps.length - 1)
        : Math.max(currentStepIndex - 1, 0);
    if (next === currentStepIndex) return;

    setIsAnimating(true);
    setSlideDir(direction === "next" ? "slide-left" : "slide-right");
    setRandomPeuImage((prev) => getRandomPeuImage(prev));

    setTimeout(() => {
      setCurrentStepIndex(next);
      setSlideDir(
        direction === "next" ? "enter-from-right" : "enter-from-left",
      );

      setTimeout(() => {
        setSlideDir("");
        setIsAnimating(false);
      }, 300);
    }, 250);
  };

  const handlePrev = () => changeStep("prev");
  const handleNext = () => changeStep("next");

  const handleRecordClick = () => {
    localStorage.setItem(
      "cookState",
      JSON.stringify({
        currentStepIndex,
        recipe,
        elapsedTime,
        voiceSessionId,
        memberId,
        dbSessionId,
        generateId,
      }),
    );

    navigate({ to: "/cook-audio" });
  };

  const handleFinishCook = () => {
    localStorage.setItem(
      "cookCompleteState",
      JSON.stringify({
        recipe,
        elapsedTime,
        dbSessionId,
        generateId,
      }),
    );

    navigate({ to: "/cook-complete" });
  };

  const formattedSteps = recipeSteps.map((step, index) => ({
    no: step.no || index + 1,
    desc: step.desc || "",
  }));

  return (
    <RecipeLayout
      steps={formattedSteps}
      currentStep={currentStepIndex + 1}
      onStepClick={(index) => {
        if (index === currentStepIndex || isAnimating) return;
        const dir = index > currentStepIndex ? "next" : "prev";
        setIsAnimating(true);
        setSlideDir(dir === "next" ? "slide-left" : "slide-right");
        setRandomPeuImage((prev) => getRandomPeuImage(prev));
        setTimeout(() => {
          setCurrentStepIndex(index);
          setSlideDir(dir === "next" ? "enter-from-right" : "enter-from-left");
          setTimeout(() => {
            setSlideDir("");
            setIsAnimating(false);
          }, 300);
        }, 250);
      }}
    >
      <div className="cook-header-row">
        <div className="cook-header-info">
          <h1 className="cook-recipe-title">{recipe.name}</h1>
          <div className="cook-time-section">
            <span className="cook-time-text">
              소요시간 {formatTime(elapsedTime)}
            </span>
          </div>
        </div>

        <div className="cook-record-section">
          <button className="cook-record-btn" onClick={handleRecordClick}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
          </button>
        </div>
      </div>

      <div className={`cook-step-box ${slideDir}`}>
        <span className="cook-step-label">
          STEP {recipeSteps[currentStepIndex]?.no || currentStepIndex + 1}
        </span>
        <p className="cook-step-description">
          {recipeSteps[currentStepIndex]?.desc || "단계 정보가 없습니다."}
        </p>
      </div>

      <div className="cook-image-nav">
        <button
          className="cook-nav-btn"
          onClick={handlePrev}
          disabled={currentStepIndex === 0 || isAnimating}
        >
          <span className="cook-arrow">‹</span>
        </button>

        <div className="cook-food-image-wrapper">
          <img
            src={randomPeuImage}
            alt="조리 이미지"
            className="cook-food-image"
            onError={(e) => {
              console.error("[CookMode] 이미지 로드 실패");
              e.target.src = peuImages[0];
            }}
          />
        </div>
        <button
          className="cook-nav-btn"
          onClick={handleNext}
          disabled={currentStepIndex === recipeSteps.length - 1 || isAnimating}
        >
          <span className="cook-arrow">›</span>
        </button>
      </div>

      {currentStepIndex !== recipeSteps.length - 1 && (
        <p className="cook-voice-hint">
          요리하시느라 손이 바쁘시죠?
          <br />
          목소리로 편하게 명령만 내려주세요!
        </p>
      )}

      {currentStepIndex === recipeSteps.length - 1 && (
        <div className="cook-finish-wrapper">
          <ButtonRed onClick={handleFinishCook}>요리 종료하기</ButtonRed>
        </div>
      )}
    </RecipeLayout>
  );
}
