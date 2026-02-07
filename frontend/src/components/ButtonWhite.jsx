// src/components/ButtonWhite.jsx
import "./ButtonWhite.css";

export default function ButtonWhite({
  onClick,
  children = "버튼",
  disabled = false,
}) {
  return (
    <button className="button-white" onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}
