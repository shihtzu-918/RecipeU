// src/components/ButtonRed.jsx
import "./ButtonRed.css";

export default function ButtonRed({
  onClick,
  children = "버튼",
  disabled = false,
  subText = null,
}) {
  return (
    <button className="button-red" onClick={onClick} disabled={disabled}>
      <span className="button-red-main">{children}</span>
      {subText && <span className="button-red-sub">{subText}</span>}
    </button>
  );
}
