// src/routes/cook-complete.jsx
import { createFileRoute } from "@tanstack/react-router";
import CookCompletePage from "@/pages/Cook/CookCompletePage";
import FixedLayout from "@/layouts/FixedLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/cook-complete")({
  component: () => (
    <FixedLayout>
      <CookCompletePage />
    </FixedLayout>
  ),
  loader: async () => {
    const images = [
      RECIPE_IMAGES["cook-complete-alert"],
      RECIPE_IMAGES["cook-potato-wink"],
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
