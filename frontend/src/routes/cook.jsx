// src/routes/cook.jsx 업데이트
import { createFileRoute } from "@tanstack/react-router";
import CookModePage from "@/pages/Cook/CookModePage";
import FixedLayout from "@/layouts/FixedLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/cook")({
  component: () => (
    <FixedLayout>
      <CookModePage />
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
