// src/routes/cook-start.jsx
import { createFileRoute } from "@tanstack/react-router";
import CookStartPage from "@/pages/Cook/CookStartPage";
import FixedLayout from "@/layouts/FixedLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/cook-start")({
  component: () => (
    <FixedLayout>
      <CookStartPage />
    </FixedLayout>
  ),
  loader: async () => {
    const images = [RECIPE_IMAGES["cook-bg-yellow"]];
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
