"use client";

import { useNavigate, useSearch, useLocation } from "@tanstack/react-router";
import React, { useEffect, useMemo, useRef, useState } from "react";
import RecipeLayout from "@/layouts/RecipeLayout";
import { RECIPE_IMAGES } from "@/images";
import "./CookModeAudioPage.css";

/**
 * CookModeAudioPage
 * - VAD로 음성 감지 → 백엔드 STT (Clova Speech) + Kiwi 완성도 분석
 * - COMPLETE → 즉시 LLM+TTS 파이프라인
 * - INCOMPLETE → 추가 대기 후 강제 전송
 * - LLM/TTS는 백엔드 SSE 스트리밍
 * - [New] Thinking Dots UI (User/AI) & Pipeline Busy Check
 */

// 백엔드 API URL
const API_URL = import.meta.env.VITE_API_URL || "";

// ====== [New] 메시지 목록 상수 ======
const WELCOME_MESSAGES = [
  "오늘의 요리, 퓨가 끝까지 옆에서 도와드릴게요. 함께 시작해볼까요?",
  "맛있는 요리가 완성될 때까지 제가 도와드릴게요. 편하게 말을 걸어주세요.",
  "요리하시느라 손이 바쁘시죠? 목소리로 편하게 명령만 내려주세요!",
  "준비되셨나요? 퓨와 함께 맛있는 요리를 만들어봐요!",
  "궁금한 점은 언제든 퓨에게 물어보세요~!",
];

const GUIDE_MESSAGES = [
  "'다 했어'라고 말하면 다음 단계로, '재료가 없어'라고 하면 대체 재료를 알려드려요!",
  "혹시 요리하다 태우거나 실수를 했나요? 당황하지 말고 퓨에게 말해주세요.",
  "다음 단계 진행을 원하시면 '다음'이라고 말해주세요.",
  "재료 대체부터 실수 해결까지, 퓨가 도와드릴게요.",
  "목소리로 요리 단계를 자유롭게 조절해보세요.",
  "재료가 없거나 막히는 부분이 생기면 언제든 퓨에게 물어보세요!",
  "오븐이 없거나 재료가 부족해도 괜찮아요. 저한테 해결 방법을 물어보세요!",
];

// ====== VAD tuning ======
const VAD_START_THRESHOLD = 0.1; // 음성 시작 감지 RMS 기준
const VAD_END_THRESHOLD = 0.025; // 음성 종료 감지 RMS 기준
const VAD_SILENCE_MS = 1500; // 침묵 대기 시간
const VAD_MIN_SPEECH_MS = 300; // 최소 발화 길이
const INCOMPLETE_EXTRA_WAIT_MS = 2000; // INCOMPLETE일 때 추가 대기

function pickMimeType() {
  const candidates = [
    "audio/mp4;codecs=mp4a.40.2",
    "audio/mp4",
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function nowTs() {
  return Date.now();
}

// ====== TTS 오디오 재생 클래스 ======
class TTSStreamPlayer {
  constructor() {
    this.audioContext = null;
    this.isPlaying = false;
    this.leftoverChunk = null;
    this.nextStartTime = 0;
    this.sampleRate = 32000;
    this.activeSources = []; // 예약된 BufferSource 추적
  }

  async init() {
    // 이전 context가 남아있으면 정리
    this.stop();
    this.audioContext = new (
      window.AudioContext || window.webkitAudioContext
    )();
    this.isPlaying = true;
    this.leftoverChunk = null;
    this.activeSources = [];
    this.nextStartTime = this.audioContext.currentTime;
  }

  setSampleRate(rate) {
    this.sampleRate = rate;
  }

  playChunk(base64Audio) {
    if (!this.audioContext || !this.isPlaying) return;

    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    let chunkToProcess = bytes;

    if (this.leftoverChunk) {
      const combined = new Uint8Array(this.leftoverChunk.length + bytes.length);
      combined.set(this.leftoverChunk);
      combined.set(bytes, this.leftoverChunk.length);
      chunkToProcess = combined;
      this.leftoverChunk = null;
    }

    const remainder = chunkToProcess.length % 2;
    if (remainder !== 0) {
      this.leftoverChunk = chunkToProcess.slice(
        chunkToProcess.length - remainder,
      );
      chunkToProcess = chunkToProcess.slice(
        0,
        chunkToProcess.length - remainder,
      );
    }

    if (chunkToProcess.byteLength === 0) return;

    const int16Data = new Int16Array(
      chunkToProcess.buffer,
      chunkToProcess.byteOffset,
      chunkToProcess.byteLength / 2,
    );
    const float32Data = new Float32Array(int16Data.length);
    for (let i = 0; i < int16Data.length; i++) {
      float32Data[i] = int16Data[i] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(
      1,
      float32Data.length,
      this.sampleRate,
    );
    audioBuffer.getChannelData(0).set(float32Data);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    if (this.nextStartTime < this.audioContext.currentTime) {
      this.nextStartTime = this.audioContext.currentTime + 0.02;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += audioBuffer.duration;

    // 재생 완료 시 목록에서 제거
    this.activeSources.push(source);
    source.onended = () => {
      this.activeSources = this.activeSources.filter((s) => s !== source);
    };
  }

  stop() {
    this.isPlaying = false;
    this.leftoverChunk = null;
    if (this.audioContext) {
      // 1) suspend로 오디오 출력 즉시 정지
      try {
        this.audioContext.suspend();
      } catch {}
      // 2) 예약된 모든 오디오 소스 중단
      for (const src of this.activeSources) {
        try {
          src.disconnect();
        } catch {}
        try {
          src.stop();
        } catch {}
      }
      // 3) context 닫기
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    this.activeSources = [];
  }
}

const ttsStreamPlayer = new TTSStreamPlayer();

export default function CookModeAudioPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const [cookState] = useState(() => {
    const saved = localStorage.getItem("cookState");
    console.log("[CookAudio] cookState 원본:", saved);

    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log("[CookAudio] 파싱된 데이터:", parsed);
        return parsed;
      } catch (err) {
        console.error("[CookAudio] 파싱 실패:", err);
      }
    }

    return {
      currentStepIndex: 0,
      elapsedTime: 0,
      recipe: {
        name: "레시피 없음",
        steps: [],
      },
    };
  });

  const passedStepIndex = cookState.currentStepIndex ?? 0;
  const passedRecipe = cookState.recipe || { name: "레시피 없음", steps: [] };
  const passedRecipeSteps = passedRecipe.steps || [];
  const passedElapsedTime = cookState.elapsedTime ?? 0;
  const passedVoiceSessionId = cookState.voiceSessionId ?? null;
  const dbSessionId = cookState.dbSessionId ?? null;
  const generateId = cookState.generateId ?? null;
  const memberId =
    cookState.memberId ??
    (() => {
      const m = localStorage.getItem("member");
      return m ? JSON.parse(m).id || 2 : 2;
    })();

  const [currentStepIndex, setCurrentStepIndex] = useState(passedStepIndex);
  const [elapsedTime, setElapsedTime] = useState(passedElapsedTime);
  const voiceSessionIdRef = useRef(passedVoiceSessionId);

  // 슬라이드 애니메이션 상태
  const [slideDir, setSlideDir] = useState("");
  const isAnimatingRef = useRef(false);

  const recipeSteps =
    passedRecipeSteps.length > 0
      ? passedRecipeSteps
      : [{ no: 1, desc: "레시피 정보가 없습니다." }];

  // 타이머
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  // Chat messages
  // [New] Feature: 최초 진입 시 랜덤 메시지 + 고정 안내 메시지 2개
  const [messages, setMessages] = useState(() => {
    const randomMsg =
      WELCOME_MESSAGES[Math.floor(Math.random() * WELCOME_MESSAGES.length)];
    return [
      { id: "welcome", type: "system", text: randomMsg, status: "done" },
      // { id: "info_1", type: "system", text: "재료 대체부터 실수 해결까지, 퓨가 도와드릴게요.", status: "done" },
      // { id: "info_2", type: "system", text: "목소리로 요리 단계를 자유롭게 조절해보세요.", status: "done" }
    ];
  });
  const [errorMsg, setErrorMsg] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [vadState, setVadState] = useState("idle");
  const [pipelineBusy, setPipelineBusy] = useState(false);

  // Pipeline Busy Ref
  const pipelineBusyRef = useRef(false);
  // 파이프라인 세부 단계: "idle" | "llm_waiting" | "tts_streaming"
  const pipelinePhaseRef = useRef("idle");

  // Thinking Message Refs
  const userThinkingMsgIdRef = useRef(null);
  const aiThinkingMsgIdRef = useRef(null);
  const pendingUserMsgIdRef = useRef(null);

  // 채팅 자동 스크롤 ref
  const chatMessagesRef = useRef(null);

  const supported = useMemo(() => {
    return !!(
      navigator?.mediaDevices?.getUserMedia &&
      typeof window !== "undefined" &&
      typeof window.MediaRecorder !== "undefined"
    );
  }, []);

  // Audio refs
  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);
  const listenTokenRef = useRef(0);
  const listenActiveRef = useRef(false);

  // VAD refs
  const vadStateRef = useRef("idle");
  const speechStartAtRef = useRef(null);
  const lastAboveAtRef = useRef(null);
  const segStartAtRef = useRef(null);

  // Recorder refs
  const segRecorderRef = useRef(null);
  const segChunksRef = useRef([]);
  const mimeTypeRef = useRef("");

  // SSE fetch 중단용
  const abortControllerRef = useRef(null);
  const isPageActiveRef = useRef(true);

  // 텍스트 버퍼
  const textBufferRef = useRef([]);
  const incompleteTimerRef = useRef(null);

  // 30초 무입력 안내 타이머
  const idleHintTimerRef = useRef(null);
  // 가이드 힌트 최대 5개까지만 표시
  const idleHintCountRef = useRef(0);
  const MAX_IDLE_HINTS = 5;

  // 30초 무입력 안내 타이머 시작/리셋
  function resetIdleHintTimer() {
    if (idleHintTimerRef.current) {
      clearTimeout(idleHintTimerRef.current);
      idleHintTimerRef.current = null;
    }
    // 최대 개수에 도달하면 더 이상 타이머 설정 안 함
    if (idleHintCountRef.current >= MAX_IDLE_HINTS) {
      return;
    }
    idleHintTimerRef.current = setTimeout(() => {
      idleHintTimerRef.current = null;
      // 가이드 메시지 중 랜덤 선택
      const randomGuide =
        GUIDE_MESSAGES[Math.floor(Math.random() * GUIDE_MESSAGES.length)];

      appendMessage({
        id: `hint_${Date.now()}`,
        type: "system",
        text: randomGuide,
        status: "done",
      });
      idleHintCountRef.current += 1;
    }, 30000);
  }

  // 채팅 메시지 변경 시 자동 스크롤
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  // currentStepIndex를 ref로도 관리
  const currentStepIndexRef = useRef(currentStepIndex);
  useEffect(() => {
    currentStepIndexRef.current = currentStepIndex;
  }, [currentStepIndex]);

  // 마이크 스트림 및 상태 정리 함수
  const cleanupAllAudio = () => {
    console.log("[cleanup] 오디오 리소스 정리 시작");
    listenActiveRef.current = false;
    listenTokenRef.current += 1;

    if (incompleteTimerRef.current) {
      clearTimeout(incompleteTimerRef.current);
      incompleteTimerRef.current = null;
    }

    if (idleHintTimerRef.current) {
      clearTimeout(idleHintTimerRef.current);
      idleHintTimerRef.current = null;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const recorder = segRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch (e) {
        console.log("[cleanup] recorder stop error:", e);
      }
      segRecorderRef.current = null;
    }

    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    const audioCtx = audioCtxRef.current;
    if (audioCtx && audioCtx.state !== "closed") {
      audioCtx.close().catch(() => {});
      audioCtxRef.current = null;
    }

    analyserRef.current = null;
    textBufferRef.current = [];

    // Thinking Refs 초기화
    pendingUserMsgIdRef.current = null;
    userThinkingMsgIdRef.current = null;
    aiThinkingMsgIdRef.current = null;
    pipelinePhaseRef.current = "idle";

    ttsStreamPlayer.stop();

    console.log("[cleanup] 오디오 리소스 정리 완료");
  };

  // 페이지 진입 시 자동으로 녹음 시작 + idle hint 타이머
  const sessionInitRef = useRef(false);
  useEffect(() => {
    isPageActiveRef.current = true;

    async function initSession() {
      // StrictMode 중복 호출 방지
      if (sessionInitRef.current) return;
      sessionInitRef.current = true;

      if (voiceSessionIdRef.current) {
        // 기존 음성 세션 기록 복원
        try {
          const res = await fetch(
            `${API_URL}/api/voice/history/${voiceSessionIdRef.current}`,
          );
          if (res.ok) {
            const data = await res.json();
            if (data?.messages?.length > 0) {
              const restored = data.messages.map((m, i) => ({
                id: `restored_${i}_${Date.now()}`,
                type: m.role.toUpperCase() === "USER" ? "user" : "ai",
                text: m.text,
                status: "done",
                restored: true,
              }));
              setMessages((prev) => [...prev, ...restored]);
              console.log("[restore] 기존 대화 복원:", restored.length, "건");
            }
          }
        } catch (e) {
          console.error("[restore] 복원 실패:", e);
        }
      } else {
        // 새 세션 생성 (비동기, 화면 블로킹 없음)
        try {
          const res = await fetch(`${API_URL}/api/voice/session`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ member_id: memberId }),
          });
          if (res.ok) {
            const data = await res.json();
            voiceSessionIdRef.current = data.session_id;
            console.log("[initSession] 새 세션 생성:", data.session_id);
          }
        } catch (e) {
          console.error("[initSession] 세션 생성 실패:", e);
        }
      }
    }

    initSession();
    startListening();
    resetIdleHintTimer();

    return () => {
      isPageActiveRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      cleanupAllAudio();
      setMessages((prev) => {
        prev.forEach((m) => {
          if (m.audioUrl) URL.revokeObjectURL(m.audioUrl);
        });
        return prev;
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 음성 대화 히스토리 저장
  async function saveVoiceHistory() {
    if (!voiceSessionIdRef.current) return;
    const chatMessages = messages
      .filter((m) => m.type === "user" || m.type === "ai")
      .filter((m) => m.status === "done" || m.status === "tts_streaming")
      .filter((m) => !m.restored)
      .map((m) => ({
        role: m.type === "user" ? "USER" : "AGENT",
        text: m.text,
      }));
    if (chatMessages.length === 0) return;
    try {
      await fetch(`${API_URL}/api/voice/save-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          member_id: memberId,
          session_id: voiceSessionIdRef.current,
          messages: chatMessages,
        }),
      });
      console.log("[saveVoiceHistory] 저장 완료:", chatMessages.length, "건");
    } catch (e) {
      console.error("[saveVoiceHistory] 저장 실패:", e);
    }
  }

  function animateStepChange(direction) {
    if (isAnimatingRef.current) return;
    isAnimatingRef.current = true;
    setSlideDir(direction === "next" ? "slide-left" : "slide-right");

    setTimeout(() => {
      setCurrentStepIndex((i) =>
        direction === "next"
          ? Math.min(i + 1, recipeSteps.length - 1)
          : Math.max(i - 1, 0),
      );
      setSlideDir(
        direction === "next" ? "enter-from-right" : "enter-from-left",
      );

      setTimeout(() => {
        setSlideDir("");
        isAnimatingRef.current = false;
      }, 300);
    }, 250);
  }

  function applyIntentAction(intent, action) {
    if (intent === "next_step") {
      if (action === "end_cooking") return;
      animateStepChange("next");
    } else if (intent === "prev_step") {
      if (action === "blocked") return;
      animateStepChange("prev");
    } else if (intent === "finish") {
      return;
    }
  }

  function appendMessage(msg) {
    const id =
      msg.id || `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    setMessages((prev) => {
      // 같은 계열(type)의 이전 thinking 메시지가 있으면 제거
      const filtered = prev.filter(
        (m) => !(m.type === msg.type && m.status === "thinking"),
      );
      return [...filtered, { id, ...msg }];
    });
    // thinking ref도 정리
    if (msg.type === "user" && msg.status !== "thinking") {
      userThinkingMsgIdRef.current = null;
    }
    if (msg.type === "ai" && msg.status !== "thinking") {
      aiThinkingMsgIdRef.current = null;
    }
    return id;
  }

  function patchMessage(id, patch) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    );
  }

  // ====== 텍스트 버퍼 → LLM+TTS 전송 ======
  function flushTextBuffer() {
    if (incompleteTimerRef.current) {
      clearTimeout(incompleteTimerRef.current);
      incompleteTimerRef.current = null;
    }

    const buffer = textBufferRef.current;
    if (buffer.length === 0) return;

    const finalText = buffer.join(" ");
    textBufferRef.current = [];

    console.log(`[flushTextBuffer] 최종 텍스트: "${finalText}"`);

    // 임시 메시지(Pending)가 있다면 확정 짓고, 없다면 새로 추가
    if (pendingUserMsgIdRef.current) {
      patchMessage(pendingUserMsgIdRef.current, {
        text: finalText,
        status: "done",
      });
      pendingUserMsgIdRef.current = null;
    } else {
      appendMessage({ type: "user", text: finalText });
    }

    // LLM+TTS 파이프라인 호출
    processTextWithBackend(finalText);
  }

  // ====== 백엔드 STT 호출 ======
  async function processSTT(audioBlob) {
    console.log("[processSTT] 호출됨, blob 크기:", audioBlob.size);

    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.webm");

    try {
      const response = await fetch(`${API_URL}/api/voice/stt`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      const { text, completeness } = result;

      if (!text) {
        console.log("[processSTT] 텍스트 없음");
        // "인식 중..." thinking 메시지 제거
        if (userThinkingMsgIdRef.current) {
          setMessages((prev) =>
            prev.filter((m) => m.id !== userThinkingMsgIdRef.current),
          );
          userThinkingMsgIdRef.current = null;
        }
        // [Bug Fix] STT가 실패(혹은 무음)했더라도 타이머를 다시 돌려야 안내가 나옴
        resetIdleHintTimer();
        return;
      }

      // 텍스트 버퍼 추가
      textBufferRef.current.push(text);
      const bufferText = textBufferRef.current.join(" ");
      console.log(`[processSTT] 인식: "${text}" → [${completeness}]`);

      // "인식 중..."(Thinking)을 실제 텍스트로 교체 (Thinking -> Pending)
      if (userThinkingMsgIdRef.current && !pendingUserMsgIdRef.current) {
        pendingUserMsgIdRef.current = userThinkingMsgIdRef.current;
        userThinkingMsgIdRef.current = null;
        patchMessage(pendingUserMsgIdRef.current, {
          text: bufferText,
          status: "pending",
        });
      } else if (pendingUserMsgIdRef.current) {
        patchMessage(pendingUserMsgIdRef.current, { text: bufferText });
      } else {
        pendingUserMsgIdRef.current = appendMessage({
          type: "user",
          text: bufferText,
          status: "pending",
        });
      }

      if (completeness === "COMPLETE") {
        console.log("[processSTT] COMPLETE → 즉시 전송");
        flushTextBuffer();
      } else {
        console.log(
          `[processSTT] INCOMPLETE → ${INCOMPLETE_EXTRA_WAIT_MS}ms 추가 대기`,
        );
        if (incompleteTimerRef.current) {
          clearTimeout(incompleteTimerRef.current);
        }
        incompleteTimerRef.current = setTimeout(() => {
          incompleteTimerRef.current = null;
          if (textBufferRef.current.length > 0) {
            console.log("[processSTT] 타임아웃! 강제 전송");
            flushTextBuffer();
          }
        }, INCOMPLETE_EXTRA_WAIT_MS);
      }
    } catch (e) {
      console.error("[processSTT] 오류:", e);
      if (userThinkingMsgIdRef.current) {
        setMessages((prev) =>
          prev.filter((m) => m.id !== userThinkingMsgIdRef.current),
        );
        userThinkingMsgIdRef.current = null;
      }
      appendMessage({
        type: "ai",
        text: "⚠️ 잠시 문제가 생겼어요. 나중에 시도해주세요!",
        status: "error",
      });
    }
  }

  async function processTextWithBackend(userText) {
    if (!isPageActiveRef.current) return;
    console.log("[processTextWithBackend] 호출됨, 텍스트:", userText);

    setPipelineBusy(true);
    pipelineBusyRef.current = true;
    pipelinePhaseRef.current = "llm_waiting";

    const thinkingId = appendMessage({
      type: "ai",
      text: "생각 중...",
      status: "thinking",
    });
    aiThinkingMsgIdRef.current = thinkingId;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const currentStepText =
      recipeSteps[currentStepIndexRef.current]?.desc ?? "";
    const currentCook = passedRecipe?.name || "";
    const currentIndex = currentStepIndexRef.current;
    const neighborIndices = [currentIndex - 1, currentIndex + 1].filter(
      (index) => index >= 0 && index < recipeSteps.length,
    );
    const neighborStepText = neighborIndices
      .map((index) => {
        const step = recipeSteps[index];
        const no = step?.no || index + 1;
        const desc = step?.desc || step?.description || "";
        return `${no}. ${desc}`.trim();
      })
      .filter(Boolean)
      .join(" / ");
    const recipeContext = neighborStepText
      ? `인접 단계: ${neighborStepText}`
      : "";

    // [요구사항 1 반영] 멀티턴 History 추출
    // - welcome 제외
    // - hint_ 로 시작하는 가이드 메시지 제외
    // - info_ 로 시작하는 초기 안내 메시지 제외
    const history = messages
      .filter((m) => m.status === "done" || m.status === "tts_streaming")
      .filter((m) => m.type === "user" || m.type === "ai")
      .filter((m) => m.status === "done" || m.status === "tts_streaming")
      .slice(-6)
      .map((m) => ({
        role: m.type === "user" ? "user" : "assistant",
        content: m.text,
      }));

    const formData = new FormData();
    formData.append("text", userText);
    formData.append("current_step", currentStepText);
    formData.append("current_cook", currentCook);
    formData.append("recipe_context", recipeContext);
    formData.append("step_index", String(currentStepIndexRef.current));
    formData.append("total_steps", String(recipeSteps.length));
    formData.append("history", JSON.stringify(history));

    let aiMsgId = null; // 실제 메시지로 변환될 ID
    let lastAction = null;
    let lastDelaySeconds = 0;

    try {
      const response = await fetch(`${API_URL}/api/voice/process-text`, {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      await ttsStreamPlayer.init();

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          try {
            const jsonStr = line.slice(6);
            const event = JSON.parse(jsonStr);

            switch (event.type) {
              case "llm":
                // LLM 응답 도착 → TTS 스트리밍 단계로 전환 (VAD 허용)
                pipelinePhaseRef.current = "tts_streaming";
                pipelineBusyRef.current = false;

                lastAction = event.action || null;
                lastDelaySeconds = event.delay_seconds || 0;

                // [UI 중요] 텍스트가 있으면 "생각 중..."을 답변으로 교체
                if (event.text) {
                  if (aiThinkingMsgIdRef.current) {
                    // Thinking 메시지를 Reuse
                    aiMsgId = aiThinkingMsgIdRef.current;
                    patchMessage(aiMsgId, {
                      text: event.text,
                      intent: event.intent,
                      status: "tts_streaming", // Thinking 해제
                    });
                    aiThinkingMsgIdRef.current = null;
                  } else {
                    // 혹시 Thinking이 없었다면 새로 생성
                    aiMsgId = appendMessage({
                      type: "ai",
                      text: event.text,
                      intent: event.intent,
                      status: "tts_streaming",
                    });
                  }
                } else {
                  // 텍스트가 없는 경우(단순 액션 등) Thinking 제거
                  if (aiThinkingMsgIdRef.current) {
                    setMessages((prev) =>
                      prev.filter((m) => m.id !== aiThinkingMsgIdRef.current),
                    );
                    aiThinkingMsgIdRef.current = null;
                  }
                }

                applyIntentAction(event.intent, event.action);
                break;

              case "tts_chunk":
                if (!isPageActiveRef.current) break;
                if (event.sample_rate) {
                  ttsStreamPlayer.setSampleRate(event.sample_rate);
                }
                ttsStreamPlayer.playChunk(event.audio);
                break;

              case "done":
                pipelinePhaseRef.current = "idle";
                if (aiMsgId) {
                  patchMessage(aiMsgId, { status: "done" });
                }
                // LLM+TTS 완료 → 10초 무입력 안내 타이머 시작
                if (lastAction !== "end_cooking" && lastAction !== "finish") {
                  resetIdleHintTimer();
                }
                if (lastAction === "end_cooking") {
                  isPageActiveRef.current = false;
                  // 즉시 VAD/마이크 중지 (타이머 전에)
                  listenActiveRef.current = false;
                  if (rafRef.current) {
                    cancelAnimationFrame(rafRef.current);
                    rafRef.current = null;
                  }
                  if (streamRef.current) {
                    streamRef.current.getTracks().forEach((t) => t.stop());
                  }
                  if (abortControllerRef.current) {
                    abortControllerRef.current.abort();
                    abortControllerRef.current = null;
                  }
                  const delay = (lastDelaySeconds || 3) * 1000;
                  setTimeout(async () => {
                    await saveVoiceHistory();
                    cleanupAllAudio();
                    localStorage.setItem(
                      "cookState",
                      JSON.stringify({
                        recipe: passedRecipe,
                        currentStepIndex: currentStepIndexRef.current,
                        elapsedTime,
                        cookingFinished: true,
                        voiceSessionId: voiceSessionIdRef.current,
                        memberId,
                        dbSessionId,
                        generateId,
                      }),
                    );
                    navigate({ to: "/cook" });
                  }, delay);
                } else if (lastAction === "finish") {
                  isPageActiveRef.current = false;
                  // 즉시 VAD/마이크 중지 (타이머 전에)
                  listenActiveRef.current = false;
                  if (rafRef.current) {
                    cancelAnimationFrame(rafRef.current);
                    rafRef.current = null;
                  }
                  if (streamRef.current) {
                    streamRef.current.getTracks().forEach((t) => t.stop());
                  }
                  if (abortControllerRef.current) {
                    abortControllerRef.current.abort();
                    abortControllerRef.current = null;
                  }
                  setIsListening(false);
                  setVadStateInternal("idle");

                  const delay = (lastDelaySeconds || 3) * 1000;
                  setTimeout(async () => {
                    await saveVoiceHistory();
                    cleanupAllAudio();
                    localStorage.setItem(
                      "cookState",
                      JSON.stringify({
                        recipe: passedRecipe,
                        currentStepIndex: currentStepIndexRef.current,
                        elapsedTime,
                        voiceSessionId: voiceSessionIdRef.current,
                        memberId,
                        dbSessionId,
                        generateId,
                      }),
                    );
                    navigate({ to: "/cook" });
                  }, delay);
                }
                break;

              case "error":
                console.error("[SSE] 서버 에러:", event.message);
                if (aiThinkingMsgIdRef.current) {
                  setMessages((prev) =>
                    prev.filter((m) => m.id !== aiThinkingMsgIdRef.current),
                  );
                  aiThinkingMsgIdRef.current = null;
                }
                appendMessage({
                  type: "ai",
                  text: "⚠️ 잠시 문제가 생겼어요. 나중에 시도해주세요!",
                  status: "error",
                });
                break;
            }
          } catch (parseErr) {
            console.error("[SSE] 파싱 오류:", parseErr, line);
          }
        }
      }
    } catch (e) {
      if (e.name === "AbortError") {
        console.log("[processTextWithBackend] 요청 중단됨 (abort)");
        return;
      }
      console.error("[processTextWithBackend] 오류:", e);
      if (aiThinkingMsgIdRef.current) {
        setMessages((prev) =>
          prev.filter((m) => m.id !== aiThinkingMsgIdRef.current),
        );
        aiThinkingMsgIdRef.current = null;
      }
      appendMessage({
        type: "ai",
        text: "⚠️ 잠시 문제가 생겼어요. 나중에 시도해주세요!",
        status: "error",
      });
    } finally {
      setPipelineBusy(false);
      pipelineBusyRef.current = false;
      pipelinePhaseRef.current = "idle";
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
    }
  }

  // ====== VAD 및 녹음 로직 ======
  function startSegmentRecording() {
    const stream = streamRef.current;
    if (!stream) return;

    segChunksRef.current = [];
    const mt = mimeTypeRef.current;
    const recorder = new MediaRecorder(
      stream,
      mt && mt !== "browser-default" ? { mimeType: mt } : undefined,
    );
    segRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) segChunksRef.current.push(e.data);
    };

    recorder.onerror = (e) => {
      console.error("[recorder] 녹음 오류:", e?.error?.message || "unknown");
    };

    recorder.onstop = async () => {
      const chunks = segChunksRef.current;
      segChunksRef.current = [];
      segRecorderRef.current = null;

      if (!chunks.length) {
        segStartAtRef.current = null;
        // "인식 중..." thinking 메시지 제거
        if (userThinkingMsgIdRef.current) {
          setMessages((prev) =>
            prev.filter((m) => m.id !== userThinkingMsgIdRef.current),
          );
          userThinkingMsgIdRef.current = null;
        }
        return;
      }

      const audioBlob = new Blob(chunks, {
        type: recorder.mimeType || "audio/webm",
      });
      const endAt = nowTs();
      const startAt = segStartAtRef.current || endAt;
      const durationMs = Math.max(0, endAt - startAt);

      if (durationMs < VAD_MIN_SPEECH_MS) {
        segStartAtRef.current = null;
        if (userThinkingMsgIdRef.current) {
          setMessages((prev) =>
            prev.filter((m) => m.id !== userThinkingMsgIdRef.current),
          );
          userThinkingMsgIdRef.current = null;
        }
        return;
      }

      segStartAtRef.current = null;
      console.log("[recorder.onstop] 오디오 생성 완료 → STT 호출");
      processSTT(audioBlob);
    };

    recorder.start(250);
  }

  function stopSegmentRecording() {
    try {
      const r = segRecorderRef.current;
      if (r && r.state !== "inactive") {
        try {
          r.requestData?.();
        } catch {}
        r.stop();
      }
    } catch {}
  }

  function setVadStateInternal(next) {
    vadStateRef.current = next;
    setVadState(next);
  }

  function loopVAD() {
    const analyser = analyserRef.current;
    if (!analyser) return;

    if (pipelinePhaseRef.current === "llm_waiting") {
      rafRef.current = requestAnimationFrame(loopVAD);
      return;
    }

    const bufferLen = analyser.fftSize;
    const data = new Uint8Array(bufferLen);
    analyser.getByteTimeDomainData(data);

    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / data.length);
    const now = nowTs();

    if (vadStateRef.current === "idle") {
      if (rms >= VAD_START_THRESHOLD) {
        console.log("[VAD] Speaking 전환");

        if (
          pipelinePhaseRef.current === "tts_streaming" ||
          ttsStreamPlayer.activeSources.length > 0
        ) {
          console.log("[VAD] 재생 중 음성 감지 → 강제 중단 및 SSE 취소");
          ttsStreamPlayer.stop();

          if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
          }

          pipelinePhaseRef.current = "idle";
          setPipelineBusy(false);
          pipelineBusyRef.current = false;
        }

        if (incompleteTimerRef.current) {
          clearTimeout(incompleteTimerRef.current);
          incompleteTimerRef.current = null;
        }

        lastAboveAtRef.current = now;
        speechStartAtRef.current = now;
        segStartAtRef.current = now;
        setVadStateInternal("speaking");

        if (!userThinkingMsgIdRef.current) {
          userThinkingMsgIdRef.current = appendMessage({
            type: "user",
            text: "인식 중...",
            status: "thinking",
          });
        }

        startSegmentRecording();
      }
    } else if (vadStateRef.current === "speaking") {
      if (rms >= VAD_END_THRESHOLD) {
        lastAboveAtRef.current = now;
      } else {
        const silentFor = now - (lastAboveAtRef.current || now);
        if (silentFor >= VAD_SILENCE_MS) {
          console.log("[VAD] Idle 전환");
          setVadStateInternal("idle");
          stopSegmentRecording();
        }
      }
    }

    rafRef.current = requestAnimationFrame(loopVAD);
  }

  async function startListening() {
    if (!supported) {
      setErrorMsg("이 브라우저는 마이크를 지원하지 않아요.");
      return;
    }

    if (streamRef.current) {
      cleanupAllAudio();
    }
    setErrorMsg("");

    listenActiveRef.current = true;
    const listenToken = listenTokenRef.current + 1;
    listenTokenRef.current = listenToken;

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      console.error("[startListening] getUserMedia error:", e);
      listenActiveRef.current = false;
      setIsListening(false);
      setErrorMsg("마이크 권한을 확인해주세요.");
      return;
    }

    if (!listenActiveRef.current || listenTokenRef.current !== listenToken) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    streamRef.current = stream;

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = audioCtx;

    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyserRef.current = analyser;

    source.connect(analyser);

    mimeTypeRef.current = pickMimeType() || "browser-default";
    lastAboveAtRef.current = null;
    speechStartAtRef.current = null;
    segStartAtRef.current = null;
    textBufferRef.current = [];
    setVadStateInternal("idle");

    setIsListening(true);
    rafRef.current = requestAnimationFrame(loopVAD);
  }

  // src/pages/Cook/CookModeAudioPage.jsx

  async function handleMicClick() {
    setIsListening(false);
    setVadStateInternal("idle");

    await saveVoiceHistory();

    cleanupAllAudio();

    localStorage.setItem(
      "cookState",
      JSON.stringify({
        recipe: passedRecipe,
        currentStepIndex,
        elapsedTime,
        voiceSessionId: voiceSessionIdRef.current,
        memberId,
        dbSessionId,
        generateId,
      }),
    );

    navigate({ to: "/cook" });
  }

  const formattedSteps = recipeSteps.map((step, index) => ({
    no: step.no || index + 1,
    desc: step.desc || "",
  }));

  return (
    <RecipeLayout
      steps={formattedSteps}
      currentStep={currentStepIndex + 1}
      onStepClick={(index) => {
        if (index === currentStepIndex || isAnimatingRef.current) return;
        const dir = index > currentStepIndex ? "next" : "prev";
        isAnimatingRef.current = true;
        setSlideDir(dir === "next" ? "slide-left" : "slide-right");
        setTimeout(() => {
          setCurrentStepIndex(index);
          setSlideDir(dir === "next" ? "enter-from-right" : "enter-from-left");
          setTimeout(() => {
            setSlideDir("");
            isAnimatingRef.current = false;
          }, 300);
        }, 250);
      }}
    >
      {/* 제목 + 소요시간 (왼쪽 6) | 녹음 버튼 (오른쪽 4) */}
      <div className="cook-header-row">
        <div className="cook-header-info">
          <h1 className="cook-recipe-title">{passedRecipe.name}</h1>
          <div className="cook-time-section">
            <span className="cook-time-text">
              소요시간 {formatTime(elapsedTime)}
            </span>
          </div>
        </div>

        <div className="cook-record-section">
          <button
            className={`cook-record-btn ${isListening ? "recording" : ""}`}
            onClick={handleMicClick}
            disabled={!supported}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
          </button>
        </div>
      </div>

      <div className={`cook-step-box ${slideDir}`}>
        <span className="cook-step-label">
          STEP {recipeSteps[currentStepIndex]?.no || currentStepIndex + 1}
        </span>
        <p className="cook-step-description">
          {recipeSteps[currentStepIndex]?.desc || "단계 정보가 없습니다."}
        </p>
      </div>

      {/* 채팅 박스 */}
      <div className="audio-chat-box">
        <div className="audio-chat-messages" ref={chatMessagesRef}>
          {/* 캐릭터 헤더 (이미지 + 이름) */}
          <div className="audio-chat-header">
            <img
              src={RECIPE_IMAGES["cook-peu-image"]}
              alt="퓨"
              className="audio-chat-avatar"
            />
            <span className="audio-chat-name">퓨</span>
          </div>

          {messages.map((msg) => (
            <div key={msg.id} className={`audio-chat-bubble ${msg.type}`}>
              <div
                className="audio-bubble-content"
                style={msg.id === "welcome" ? { fontWeight: "700" } : {}}
              >
                {msg.status === "thinking" ? (
                  <>
                    <div className="thinking-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <span>{msg.text}</span>
                  </>
                ) : (
                  msg.text
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {errorMsg && <div className="audio-error-msg">{errorMsg}</div>}
    </RecipeLayout>
  );
}
