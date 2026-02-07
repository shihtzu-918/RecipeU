// src/routes/home.jsx
import { createFileRoute } from "@tanstack/react-router";
import HomePage from "@/pages/Home/HomePage";
import ScrollLayout from "@/layouts/ScrollLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/home")({
  component: () => (
    <ScrollLayout>
      <HomePage />
    </ScrollLayout>
  ),
  loader: async () => {
    const images = [
      RECIPE_IMAGES["main-bg"],
      RECIPE_IMAGES["main-character"],
      RECIPE_IMAGES["birthday-main-character_v2"],
      RECIPE_IMAGES["main-profile"],
      RECIPE_IMAGES["main-weather"],
      RECIPE_IMAGES["main-next"],
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
