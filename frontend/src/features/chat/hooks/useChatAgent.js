// src/features/chat/hooks/useChatAgent.js
import { useState, useEffect, useRef } from "react";

export const useChatAgent = (sessionId) => {
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [currentProgress, setCurrentProgress] = useState("");
  const [lastRecommendation, setLastRecommendation] = useState(null);

  const wsRef = useRef(null);
  const API_URL = import.meta.env.VITE_API_URL || "http://211.188.62.72:8080";
  const WS_URL = import.meta.env.VITE_WS_URL || "ws://211.188.62.72:8080";

  useEffect(() => {
    // WebSocket 연결
    const ws = new WebSocket(`${WS_URL}/api/chat/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[ChatAgent] Connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("[ChatAgent] Received:", data);

      if (data.type === "agent_message") {
        // Agent 응답 메시지
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.content,
            recipeInfo: data.recipe_info,
            timestamp: new Date().toISOString(),
          },
        ]);

        if (data.recipe_info) {
          setLastRecommendation(data.recipe_info);
        }

        setIsThinking(false);
        setCurrentProgress("");
      } else if (data.type === "thinking") {
        setIsThinking(true);
        setCurrentProgress("생각 중...");
      } else if (data.type === "progress") {
        setCurrentProgress(data.message);
      } else if (data.type === "error") {
        console.error("Chat Agent Error:", data.message);
        alert(data.message);
        setIsThinking(false);
      }
    };

    ws.onclose = () => {
      console.log("[ChatAgent] Disconnected");
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("[ChatAgent] Error:", error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const sendMessage = (content) => {
    if (!wsRef.current || !isConnected) {
      console.error("WebSocket not connected");
      return;
    }

    // 사용자 메시지 추가
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      },
    ]);

    // 서버로 전송
    wsRef.current.send(
      JSON.stringify({
        type: "user_message",
        content,
      }),
    );

    setIsThinking(true);
  };

  return {
    messages,
    isConnected,
    isThinking,
    currentProgress,
    lastRecommendation,
    sendMessage,
  };
};
