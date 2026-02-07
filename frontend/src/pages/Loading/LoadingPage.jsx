// src/pages/Loading/LoadingPage.jsx
import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "@tanstack/react-router";
import "./LoadingPage.css";
import { RECIPE_IMAGES } from "@/images";

const tips = [
  '"설탕이 몇 스푼이었지?"라고 물어보세요.\n지나간 단계의 계량도 바로 알려드려요.',
  '"앗, 올리고당이 없네?"\n당황하지 말고 "설탕 대신 뭘 넣을까?"라고 물어보세요.',
  '"방금 뭐라고 했어?"라고 하면 다시 들려드려요.',
  '"적당히 볶는 게 어느 정도야?"라고 물어보세요.\n요리 초보를 위한 꿀팁을 드려요.',
  '"잠시만 멈춰줘"라고 말하세요.\n당신의 속도에 맞출게요.',
  '"계량 스푼이 없는데 어떡해?"라고 물어보세요.\n밥숟가락 기준의 계량을 알려드려요.',
  '"아까 넣은 간장이랑 지금 넣는 게 같은 거야?"\n같은 까다로운 질문도 이해해요.',
  '"다음 단계 재료가 뭐야?"라고 미리 물어보고\n냉장고에서 꺼내오세요.',
  '"찌개가 너무 짠데 어떡하지?"같은\n돌발 상황 해결법도 레시퓨는 알고 있어요.',
  '"이거 얼마나 더 끓여야 돼?"라고 물어보세요.\n남은 조리 시간을 체크해 드릴게요.',
];

function BounceText({ text }) {
  return (
    <span>
      {[...text].map((char, i) => (
        <span
          key={i}
          className="bounce-char"
          style={{ animationDelay: `${i * 0.07}s` }}
        >
          {char === " " ? "\u00A0" : char}
        </span>
      ))}
    </span>
  );
}

export default function LoadingPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // currentTipIndex 상태 선언
  const [currentTipIndex, setCurrentTipIndex] = useState(0);

  const { memberInfo, chatHistory, sessionId, isRegeneration } =
    location.state || {};

  const API_URL = import.meta.env.VITE_API_URL || "";

  // 7초마다 랜덤 팁 변경
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTipIndex(Math.floor(Math.random() * tips.length));
    }, 7000);
    return () => clearInterval(interval);
  }, []);

  // 레시피 생성 API 호출
  useEffect(() => {
    let isActive = true;
    const abortController = new AbortController();

    const generateRecipe = async () => {
      if (!sessionId) {
        alert("세션 정보가 없습니다.");
        if (isActive) {
          navigate({ to: "/chat" });
        }
        return;
      }

      try {
        console.log("[LoadingPage] 레시피 생성 요청:", sessionId);

        const response = await fetch(
          `${API_URL}/api/recipe/generate-from-chat?session_id=${sessionId}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            signal: abortController.signal,
          },
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error("[LoadingPage] 에러 응답:", errorData);
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("[LoadingPage] 레시피 생성 완료:", data);

        const imageUrl = data.recipe?.image || data.recipe?.img_url || "";

        console.log("[LoadingPage] 이미지 URL:", imageUrl);

        if (isActive) {
          navigate({
            to: "/recipe-result",
            state: {
              recipe: data.recipe,
              userId: data.user_id || data.member_id,
              title: data.title,
              constraints: data.constraints,
              sessionId: sessionId,
              dbSessionId: data.db_session_id,
              generateId: data.generate_id,
              memberInfo: memberInfo,
              chatHistory: chatHistory,
              imageUrl: imageUrl,
              remainingCount: isRegeneration ? 0 : 1,
            },
            replace: true,
          });
        }
      } catch (error) {
        if (!isActive || error.name === "AbortError") {
          return;
        }
        console.error("[LoadingPage] 레시피 생성 실패:", error);
        alert("레시피 생성에 실패했습니다. 다시 시도해주세요.");
        navigate({ to: "/chat", replace: true });
      }
    };

    generateRecipe();
    return () => {
      isActive = false;
      abortController.abort();
    };
  }, [API_URL, sessionId, memberInfo, chatHistory, isRegeneration, navigate]);

  return (
    <div className="loading-page">
      <div
        className="home-bg"
        style={{ backgroundImage: `url(${RECIPE_IMAGES["main-bg"]})` }}
      />

      <div className="loading-text">
        <h2>
          <BounceText text="맞춤 레시피를 생성하고 있어요" />
        </h2>
        <p>
          <BounceText text="잠시만 기다려주세요..." />
        </p>
      </div>

      <div className="phone-wrapper">
        <img
          src="./loading-motion.gif"
          alt="로딩 애니메이션"
          className="loading-gif"
        />
        <img
          src="./loading-bg-phone.png"
          alt="폰 프레임"
          className="phone-frame"
        />
      </div>

      <div className="loading-tips">
        <div className="tip-item" key={currentTipIndex}>
          <span className="tip-icon">💡</span>
          <p>{tips[currentTipIndex]}</p>
        </div>
      </div>
    </div>
  );
}
