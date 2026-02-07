// src/routes/recipes.my.jsx

import { createFileRoute } from "@tanstack/react-router";
import MyRecipesPage from "@/pages/MyRecipes/MyRecipesPage";
import ScrollLayout from "@/layouts/ScrollLayout";
import { RECIPE_IMAGES } from "@/images";

const API_URL = import.meta.env.VITE_API_URL || "";

export const Route = createFileRoute("/recipes/my")({
  component: () => (
    <ScrollLayout>
      <MyRecipesPage />
    </ScrollLayout>
  ),

  loader: async () => {
    // ✅ 캐시 확인
    const cached = localStorage.getItem("recipesMyCache");
    if (cached) {
      try {
        const { data, timestamp } = JSON.parse(cached);
        const FIVE_MINUTES = 5 * 60 * 1000;

        if (Date.now() - timestamp < FIVE_MINUTES) {
          console.log("[MyRecipes] 캐시된 데이터 사용");

          // 이미지만 추가 로드
          const images = [
            RECIPE_IMAGES["cook-bg-brown"],
            RECIPE_IMAGES["my-recipe-clip-beige"],
            RECIPE_IMAGES["my-recipe-board"],
          ];
          await Promise.all(
            images.map(
              (src) =>
                new Promise((resolve) => {
                  const img = new Image();
                  img.onload = img.onerror = resolve;
                  img.src = src;
                }),
            ),
          );

          return data;
        }
      } catch (err) {
        console.error("[MyRecipes] 캐시 파싱 실패:", err);
      }
    }

    // ✅ 캐시 없으면 새로 fetch
    console.log("[MyRecipes] 새로 데이터 가져오기");
    const images = [
      RECIPE_IMAGES["cook-bg-brown"],
      RECIPE_IMAGES["my-recipe-clip-beige"],
      RECIPE_IMAGES["my-recipe-board"],
    ];

    const imagePromises = images.map(
      (src) =>
        new Promise((resolve) => {
          const img = new Image();
          img.onload = img.onerror = resolve;
          img.src = src;
        }),
    );

    const memberStr = localStorage.getItem("member");
    const member = memberStr ? JSON.parse(memberStr) : null;
    const memberId = member?.id || 0;

    const dataPromise = fetch(
      `${API_URL}/api/recipe/list?member_id=${memberId}`,
    )
      .then((res) => res.json())
      .catch(() => ({ recipes: [] }));

    const [, data] = await Promise.all([
      Promise.all(imagePromises),
      dataPromise,
    ]);
    return data;
  },
});
