// src/pages/Chat/ChatPageOut.jsx
import { useState, useEffect, useRef } from "react";
import { useNavigate, useRouter } from "@tanstack/react-router";
import { RECIPE_IMAGES } from "@/images";
import { formatMarkdown } from "@/utils/textFormatter";
import "./ChatPageOut.css";

// 볼드 포맷팅 함수
function formatBoldText(text) {
  return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

export default function ChatPageOut() {
  const navigate = useNavigate();
  const router = useRouter();
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "안녕하세요! 무엇이 궁금하신가요?",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  const wsRef = useRef(null);
  const sessionIdRef = useRef(crypto.randomUUID());
  const sessionId = sessionIdRef.current;
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const WS_URL = import.meta.env.VITE_WS_URL || "";

  useEffect(() => {
    if (!isThinking && isConnected) {
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  }, [isThinking, isConnected]);

  // 스크롤 최하단
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  // WebSocket 연결
  useEffect(() => {
    console.log("[External WS] 연결 시작...");
    const ws = new WebSocket(`${WS_URL}/api/chat-external/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[External WS] Connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("[External WS] Received:", data);

      if (data.type === "assistant_message") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.content,
            timestamp: new Date().toISOString(),
          },
        ]);
        setIsThinking(false);
      } else if (data.type === "thinking") {
        setIsThinking(true);
      } else if (data.type === "error") {
        console.error("Error:", data.message);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.message || "오류가 발생했습니다.",
            timestamp: new Date().toISOString(),
          },
        ]);
        setIsThinking(false);
      }
    };

    ws.onclose = () => {
      console.log("[External WS] Disconnected");
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("[External WS] Error:", error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [WS_URL, sessionId]);

  // 메시지 전송
  const handleSend = () => {
    if (!input.trim() || !isConnected || isThinking) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: input,
        timestamp: new Date().toISOString(),
      },
    ]);

    wsRef.current.send(
      JSON.stringify({
        type: "user_message",
        content: input,
      }),
    );

    setInput("");
    setIsThinking(true);
  };

  // textarea 자동 높이 조절
  const handleTextareaChange = (e) => {
    setInput(e.target.value);

    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "48px";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 72)}px`;
    }
  };

  return (
    <div
      className="external-chat-page"
      style={{ backgroundImage: `url(${RECIPE_IMAGES["cook-bg-green"]})` }}
    >
      {/* 헤더 */}
      <button
        className="external-back-button"
        onClick={() => router.history.back()}
      >
        <img
          src={RECIPE_IMAGES["back-icon"]}
          alt="뒤로가기"
          className="back-icon"
        />
      </button>
      <div className="external-chat-header">
        <h1>레시퓨에게 물어보세요</h1>
      </div>

      {/* 메시지 영역 */}
      <div className="external-chat-content">
        <div className="external-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`external-message ${msg.role}`}>
              <div
                className="external-bubble"
                dangerouslySetInnerHTML={{
                  __html: formatMarkdown(msg.content),
                }}
              />
            </div>
          ))}

          {isThinking && (
            <div className="external-message assistant">
              <div className="external-bubble external-thinking">
                <div className="external-thinking-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span>생각 중...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* 입력창 */}
      <div className="external-input-area">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleTextareaChange}
          onKeyPress={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder={isConnected ? "무엇이든 물어보세요..." : "연결 중..."}
          disabled={!isConnected || isThinking}
          rows={1}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || !isConnected || isThinking}
        >
          전송
        </button>
      </div>
    </div>
  );
}
