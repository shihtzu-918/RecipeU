// src/routes/__root.jsx
import { createRootRoute, Outlet } from "@tanstack/react-router";
import MobileLayout from "@/layouts/MobileLayout";

export const Route = createRootRoute({
  component: () => (
    <MobileLayout>
      <Outlet />
    </MobileLayout>
  ),
});
