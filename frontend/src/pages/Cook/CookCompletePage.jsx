// src/pages/Cook/CookCompletePage.jsx

"use client";

import { useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import RecipeLayout from "@/layouts/RecipeLayout";
import { RECIPE_IMAGES } from "@/images";
import "./CookCompletePage.css";

export default function CookCompletePage() {
  const navigate = useNavigate();

  // localStorage에서 cookCompleteState 복원
  const [completeState] = useState(() => {
    const saved = localStorage.getItem("cookCompleteState");
    console.log("[CookComplete] cookCompleteState:", saved);

    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log("[CookComplete] 파싱된 데이터:", parsed);
        return parsed;
      } catch (err) {
        console.error("[CookComplete] 파싱 실패:", err);
      }
    }

    return {
      recipe: {
        name: "레시피 없음",
        image: RECIPE_IMAGES["default_img"],
        ingredients: [],
        steps: [],
      },
      elapsedTime: 0,
    };
  });

  const recipe = completeState.recipe;
  const elapsedTime = completeState.elapsedTime;
  const dbSessionId = completeState.dbSessionId ?? null;
  const generateId = completeState.generateId ?? null;

  console.log("[CookComplete] dbSessionId:", dbSessionId);
  console.log("[CookComplete] generateId:", generateId);

  // ✅ 마운트 시 데이터 확인 및 리다이렉트
  useEffect(() => {
    console.log("[CookComplete] 레시피:", recipe);
    console.log("[CookComplete] 재료:", recipe.ingredients);
    console.log("[CookComplete] 조리 단계:", recipe.steps);

    if (!recipe || !recipe.name || recipe.name === "레시피 없음") {
      console.error("[CookComplete] 레시피 데이터 없음");
      alert("레시피 데이터를 불러올 수 없습니다.");
      navigate({ to: "/home" });
    }
  }, [recipe, navigate]);

  const [rating, setRating] = useState(2);
  const [saveStatus, setSaveStatus] = useState(null);
  const [isSaved, setIsSaved] = useState(false);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  const API_URL = import.meta.env.VITE_API_URL || "";

  const memberStr = localStorage.getItem("member");
  const member = memberStr ? JSON.parse(memberStr) : null;
  const memberId = member?.id || 0;

  const handleSaveRecipe = async () => {
    try {
      // ✅ 변수 정의
      const name = recipe.name || recipe.title || "제목 없음";
      const title = name;
      const image = recipe.image || "";
      const ingredients = recipe.ingredients || [];
      const steps = recipe.steps || [];
      const cook_time = recipe.time || recipe.cook_time || "30분";
      const level = recipe.level || "중급";

      const payload = {
        user_id: memberId,
        recipe: {
          name,
          title,
          image,
          ingredients,
          steps,
          cook_time,
          level,
        },
        rating: rating,
        generate_id: generateId,
        session_id: dbSessionId,
        elapsed_time: elapsedTime,
      };

      console.log(
        "[CookComplete] 저장 요청:",
        JSON.stringify(payload, null, 2),
      );

      const response = await fetch(`${API_URL}/api/recipe/save-my-recipe`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("[CookComplete] 저장 성공:", data);
        setSaveStatus("success");
        setIsSaved(true);
        setTimeout(() => setSaveStatus(null), 2500);
      } else {
        const errorText = await response.text();
        console.error("[CookComplete] 저장 실패:", errorText);
        setSaveStatus("fail");
        setTimeout(() => setSaveStatus(null), 2500);
      }
    } catch (error) {
      console.error("[CookComplete] 저장 에러:", error);
      setSaveStatus("fail");
      setTimeout(() => setSaveStatus(null), 2500);
    }
  };

  const handleSkip = () => {
    // 홈으로 가기 전 localStorage 정리
    localStorage.removeItem("cookCompleteState");
    localStorage.removeItem("cookState");
    navigate({ to: "/home" });
  };

  return (
    <RecipeLayout steps={[]} showBottomSheet={false}>
      <div className="complete-title-section">
        <h1 className="complete-title">오늘의 요리가 끝났어요</h1>
        <p className="complete-subtitle">레시피를 전달드릴게요</p>
      </div>

      <div className="complete-recipe-card">
        <h2 className="complete-recipe-name">{recipe.name}</h2>
        <p className="complete-recipe-time">
          총 소요시간 {formatTime(elapsedTime)}
        </p>
      </div>

      <div className="complete-food-image-wrapper">
        <img
          src={recipe.image || "/default-food.jpg"}
          alt={recipe.name}
          className="complete-food-image"
          onError={(e) => {
            e.target.src = "/default-food.jpg";
          }}
        />
        {saveStatus && (
          <div
            className={`complete-saved-toast ${saveStatus === "fail" ? "fail" : ""}`}
          >
            {saveStatus === "success" && (
              <img
                src={RECIPE_IMAGES["cook-complete-alert"]}
                alt="완료"
                className="complete-saved-icon"
              />
            )}
            <span className="complete-saved-text">
              {saveStatus === "success" ? "담기 완료!" : "저장 실패"}
            </span>
          </div>
        )}
      </div>

      <div className="complete-rating">
        {[1, 2, 3].map((star) => (
          <button
            key={star}
            className={`star-btn ${star <= rating ? "filled" : ""}`}
            onClick={() => setRating(star)}
          >
            {star <= rating ? "★" : "☆"}
          </button>
        ))}
      </div>

      <div className="complete-buttons">
        {memberId !== 0 ? (
          <button
            className="btn-save"
            onClick={handleSaveRecipe}
            disabled={isSaved}
            style={isSaved ? { opacity: 0.5, cursor: "not-allowed" } : {}}
          >
            {isSaved ? "담기 완료!" : "마이레시피에"}
            <br />
            {isSaved ? "" : "담을래요"}
          </button>
        ) : (
          <button className="btn-save" disabled style={{ opacity: 0.5 }}>
            저장 불가
            <br />
            (로그인 필요)
          </button>
        )}
        <button
          className="btn-skip"
          onClick={handleSkip}
          disabled={isSaved}
          style={isSaved ? { opacity: 0.5, cursor: "not-allowed" } : {}}
        >
          안담을래요
        </button>
      </div>
    </RecipeLayout>
  );
}
