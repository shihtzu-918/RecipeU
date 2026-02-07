import { createFileRoute } from "@tanstack/react-router";
import RankingRecipesPage from "@/pages/Rankings/RankingRecipesPage";
import ScrollLayout from "@/layouts/ScrollLayout";
import { RECIPE_IMAGES } from "@/images";
// src/routes/recipes.index.jsx

export const Route = createFileRoute("/recipes/")({
  component: () => (
    <ScrollLayout>
      <RankingRecipesPage />
    </ScrollLayout>
  ),
  loader: async () => {
    const API_URL = import.meta.env.VITE_API_URL || "";

    // 캐시 확인
    const cached = localStorage.getItem("recipesRankingCache");
    if (cached) {
      try {
        const { data, timestamp } = JSON.parse(cached);
        const FIVE_MINUTES = 5 * 60 * 1000;

        // 5분 이내면 캐시 사용
        if (Date.now() - timestamp < FIVE_MINUTES) {
          console.log("[Recipes] 캐시된 데이터 사용");

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
        console.error("[Recipes] 캐시 파싱 실패:", err);
      }
    }

    // 캐시 없으면 새로 fetch
    console.log("[Recipes] 새로 데이터 가져오기");
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

    const dataPromise = fetch(`${API_URL}/api/rankings/today?limit=100`)
      .then((res) => res.json())
      .catch(() => ({ recipes: [] }));

    const [, data] = await Promise.all([
      Promise.all(imagePromises),
      dataPromise,
    ]);

    return data;
  },
});
