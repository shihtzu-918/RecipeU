// src/components/PageTransition.jsx
import { useState, useEffect } from "react";
import { useLocation } from "@tanstack/react-router";

export default function PageTransition({
  children,
  images = [],
  fetchData = null,
}) {
  const [isReady, setIsReady] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setIsReady(false);

    const loadResources = async () => {
      try {
        // 이미지 로드
        const imagePromises = images.map((src) => {
          return new Promise((resolve) => {
            const img = new Image();
            img.onload = img.onerror = resolve;
            img.src = src;
          });
        });

        // 데이터 fetch
        const dataPromise = fetchData ? fetchData() : Promise.resolve();

        // 모든 리소스 로드 대기
        await Promise.all([...imagePromises, dataPromise]);

        setIsReady(true);
      } catch (error) {
        console.error("Resource loading error:", error);
        setIsReady(true); // 에러 나도 페이지 표시
      }
    };

    loadResources();
  }, [location.pathname, images, fetchData]);

  if (!isReady) {
    return null; // 빈 화면 (또는 로딩 중)
  }

  return <div className="page-fade-in">{children}</div>;
}
