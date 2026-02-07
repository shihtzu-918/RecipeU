// src/routes/chat.jsx
import { createFileRoute } from "@tanstack/react-router";
import ChatPage from "@/pages/Chat/ChatPage";
import FixedLayout from "@/layouts/FixedLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/chat")({
  component: () => (
    <FixedLayout>
      <ChatPage />
    </FixedLayout>
  ),
  validateSearch: (search) => ({
    sessionId: search.sessionId,
    skipToChat: search.skipToChat,
    fromRegenerate: search.fromRegenerate,
  }),
  loader: async () => {
    const images = [RECIPE_IMAGES["cook-bg-green"]];
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
