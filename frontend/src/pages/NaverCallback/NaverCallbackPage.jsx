import { useEffect, useState } from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";

const API_URL = import.meta.env.VITE_API_URL || "";

export default function NaverCallbackPage() {
  const navigate = useNavigate();
  const searchParams = useSearch({ from: '/naver-callback' });
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleNaverCallback = async () => {
      const code = searchParams.code;
      const state = searchParams.state;

      if (!code || !state) {
        setError("인증 정보가 없습니다.");
        return;
      }

      try {
        const callbackUrl = `${window.location.origin}/naver-callback`;
        
        const response = await fetch(
          `${API_URL}/api/auth/callback?code=${code}&state=${state}&callback_url=${encodeURIComponent(callbackUrl)}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        const data = await response.json();

        if (response.ok) {
          // 회원 정보 저장
          localStorage.setItem("member", JSON.stringify(data.member));
          localStorage.setItem("userId", data.member.id);
          navigate({ to: "/home" });
        } else {
          setError(data.detail || "로그인 실패");
        }
      } catch (err) {
        console.error("네이버 로그인 에러:", err);
        setError("로그인 처리 중 오류가 발생했습니다.");
      }
    };

    handleNaverCallback();
  }, [searchParams, navigate]);

  if (error) {
    return (
      <div style={{ padding: "20px", textAlign: "center" }}>
        <h2>로그인 실패</h2>
        <p>{error}</p>
        <button onClick={() => navigate({ to: "/" })}>
          로그인 페이지로 돌아가기
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h2>네이버 로그인 처리 중...</h2>
    </div>
  );
}
