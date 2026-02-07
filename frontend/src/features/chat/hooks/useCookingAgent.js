// src/features/chat/hooks/useCookingAgent.js
import { useState, useEffect, useRef } from "react";

export const useCookingAgent = (sessionId, recipe) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [response, setResponse] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  const wsRef = useRef(null);
  const API_URL = import.meta.env.VITE_API_URL || "http://211.188.62.72:8080";
  const WS_URL = import.meta.env.VITE_WS_URL || "ws://211.188.62.72:8080";

  useEffect(() => {
    if (!recipe) return;

    // WebSocket 연결
    const ws = new WebSocket(`${WS_URL}/api/cook/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[CookingAgent] Connected");
      setIsConnected(true);

      // 레시피 초기화
      ws.send(
        JSON.stringify({
          type: "init",
          recipe,
        }),
      );
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "cook_response") {
        setResponse(data.text);
        setCurrentStep(data.step_index);
        setTotalSteps(data.total_steps);

        if (data.audio_url) {
          setAudioUrl(`${API_URL}${data.audio_url}`);
        }

        setIsThinking(false);
      } else if (data.type === "thinking") {
        setIsThinking(true);
      } else if (data.type === "error") {
        console.error("Cooking Agent Error:", data.message);
        alert(data.message);
        setIsThinking(false);
      }
    };

    ws.onclose = () => {
      console.log("[CookingAgent] Disconnected");
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("[CookingAgent] Error:", error);
      setIsConnected(false);
    };

    // 레시피 단계 수 설정
    if (recipe?.steps) {
      setTotalSteps(recipe.steps.length);
    }

    return () => {
      ws.close();
    };
  }, [sessionId, recipe]);

  const sendText = (text) => {
    if (!wsRef.current || !isConnected) return;

    setIsThinking(true);
    wsRef.current.send(
      JSON.stringify({
        type: "text_input",
        text,
      }),
    );
  };

  const uploadVoice = async (file) => {
    setIsThinking(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/cook/audio/${sessionId}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      setResponse(data.text);
      setCurrentStep(data.step_index);
      setTotalSteps(data.total_steps);

      if (data.audio_url) {
        setAudioUrl(`${API_URL}${data.audio_url}`);
      }
    } catch (error) {
      console.error("Voice upload failed:", error);
      throw error;
    } finally {
      setIsThinking(false);
    }
  };

  const goNext = () => sendText("다음");
  const goPrev = () => sendText("이전");

  return {
    currentStep,
    totalSteps,
    response,
    audioUrl,
    isConnected,
    isThinking,
    sendText,
    uploadVoice,
    goNext,
    goPrev,
  };
};
