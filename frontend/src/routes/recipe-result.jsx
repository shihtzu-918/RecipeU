// src/routes/recipe-result.jsx 업데이트
import { createFileRoute } from "@tanstack/react-router";
import RecipeResultPage from "@/pages/Recipes/RecipeResultPage";
import FixedLayout from "@/layouts/FixedLayout";

export const Route = createFileRoute("/recipe-result")({
  component: () => (
    <FixedLayout>
      <RecipeResultPage />
    </FixedLayout>
  ),
  validateSearch: (search) => ({
    sessionId: search.sessionId,
  }),
});
