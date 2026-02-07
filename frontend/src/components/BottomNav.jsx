import { useNavigate, useLocation } from "@tanstack/react-router";
import "./BottomNav.css";
import { RECIPE_IMAGES } from "@/images";

export default function BottomNav() {
  const navigate = useNavigate();
  const location = useLocation();

  const menus = [
    {
      path: "/home",
      label: "홈",
      activeIcon: RECIPE_IMAGES["nav-home-click"],
      inactiveIcon: RECIPE_IMAGES["nav-home-non"],
    },
    {
      path: "/cook-start",
      label: "조리모드",
      activeIcon: RECIPE_IMAGES["nav-cook-click"],
      inactiveIcon: RECIPE_IMAGES["nav-cook-non"],
    },
    {
      path: "/recipes/my",
      label: "마이 레시피",
      activeIcon: RECIPE_IMAGES["nav-my-click"],
      inactiveIcon: RECIPE_IMAGES["nav-my-non"],
    },
    {
      path: "/out-chat",
      label: "챗봇",
      activeIcon: RECIPE_IMAGES["nav-chat-click"],
      inactiveIcon: RECIPE_IMAGES["nav-chat-non"],
    },
  ];

  return (
    <nav className="bottom-nav">
      {menus.map((menu) => {
        const isActive = location.pathname === menu.path;

        return (
          <button
            key={menu.path}
            className={`nav-item ${isActive ? "active" : ""}`}
            onClick={() => navigate({ to: menu.path })}
          >
            <img
              src={isActive ? menu.activeIcon : menu.inactiveIcon}
              alt={menu.label}
            />
            <span>{menu.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
