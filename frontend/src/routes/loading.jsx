// src/routes/loading.jsx 업데이트
import { createFileRoute } from "@tanstack/react-router";
import LoadingPage from "@/pages/Loading/LoadingPage";
import FixedLayout from "@/layouts/FixedLayout";

export const Route = createFileRoute("/loading")({
  component: () => (
    <FixedLayout>
      <LoadingPage />
    </FixedLayout>
  ),
  validateSearch: (search) => ({
    sessionId: search.sessionId,
    isRegeneration: search.isRegeneration,
  }),
});
