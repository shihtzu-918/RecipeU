// src/components/RecipeBottomSheet.jsx
import { useState } from "react";
import "./RecipeBottomSheet.css";

export default function RecipeBottomSheet({
  steps,
  currentStep = 1,
  onStepClick = null, // 클릭 핸들러 (없으면 클릭 비활성화)
}) {
  const [isOpen, setIsOpen] = useState(false);

  if (!steps || steps.length === 0) {
    return null;
  }

  const handleStepClick = (index) => {
    if (onStepClick) {
      onStepClick(index);
      setIsOpen(false);
    }
  };

  return (
    <>
      {/* Bottom Sheet Overlay */}
      <div
        className={`bottom-sheet-overlay ${isOpen ? "active" : ""}`}
        onClick={() => setIsOpen(false)}
      />

      {/* Bottom Sheet - 하단 고정 */}
      <div className={`recipe-bottom-sheet ${isOpen ? "open" : ""}`}>
        {/* 항상 보이는 트리거 */}
        <div className="sheet-trigger" onClick={() => setIsOpen(!isOpen)}>
          <div className="trigger-indicator" />
          <span className="trigger-text">레시피 전체보기</span>
        </div>

        {/* 올라오는 콘텐츠 */}
        {isOpen && (
          <div className="sheet-content">
            <div className="recipe-steps-list">
              {steps.map((step, index) => {
                const stepNumber = step.no || step.id || step.step || index + 1;

                let rawText =
                  step.desc ||
                  step.text ||
                  step.description ||
                  (typeof step === "string" ? step : "") ||
                  "";

                // 앞 번호 제거
                const stepText = rawText.replace(/^\s*\d+\.\s*/, "");

                const isClickable = !!onStepClick;

                return (
                  <div
                    key={index}
                    className={`recipe-step-item ${
                      stepNumber === currentStep ? "active" : ""
                    } ${isClickable ? "clickable" : ""}`}
                    onClick={() => handleStepClick(index)}
                  >
                    <span className="step-num">{stepNumber}.</span>
                    <span className="step-desc">{stepText}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
