// src/routes/out-chat.jsx
import { createFileRoute } from "@tanstack/react-router";
import ChatPageOut from "@/pages/Chat/ChatPageOut";
import FixedLayout from "@/layouts/FixedLayout";
import { RECIPE_IMAGES } from "@/images";

export const Route = createFileRoute("/out-chat")({
  component: () => (
    <FixedLayout>
      <ChatPageOut />
    </FixedLayout>
  ),
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
