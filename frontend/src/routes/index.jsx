// src/routes/index.jsx
import { createFileRoute } from "@tanstack/react-router";
import SplashPage from "@/pages/Splash/SplashPage";
import FixedLayout from "@/layouts/FixedLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/")({
  component: () => (
    <FixedLayout>
      <SplashPage />
    </FixedLayout>
  ),
  loader: async () => {
    const images = [
      RECIPE_IMAGES["main-character"],
      RECIPE_IMAGES["birthday-main-character_v2"],
      RECIPE_IMAGES["main-bg"],
      RECIPE_IMAGES["potato-face"],
      RECIPE_IMAGES["main-profile"],
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
  },
});
