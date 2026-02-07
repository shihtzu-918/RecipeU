// src/routes/mypage.jsx
import { createFileRoute } from "@tanstack/react-router";
import MyPage from "@/pages/MyPages/MyPage";
import FixedLayout from "@/layouts/FixedLayout";

export const Route = createFileRoute("/mypage")({
  component: () => (
    <FixedLayout>
      <MyPage />
    </FixedLayout>
  ),
});
