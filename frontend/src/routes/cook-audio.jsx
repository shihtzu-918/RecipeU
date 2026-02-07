// src/routes/cook-audio.jsx
import { createFileRoute } from "@tanstack/react-router";
import CookModeAudioPage from "@/pages/Cook/CookModeAudioPage";
import FixedLayout from "@/layouts/FixedLayout";

export const Route = createFileRoute("/cook-audio")({
  component: () => (
    <FixedLayout>
      <CookModeAudioPage />
    </FixedLayout>
  ),
});
