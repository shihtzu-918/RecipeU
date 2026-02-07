export default function Button({ children, onClick, variant }) {
  return (
    <button className={`btn ${variant}`} onClick={onClick}>
      {children}
    </button>
  );
}
