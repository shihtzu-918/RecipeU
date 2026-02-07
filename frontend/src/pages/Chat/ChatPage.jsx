// src/pages/Chat/ChatPage.jsx
import { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "@tanstack/react-router";
import ButtonRed from "@/components/ButtonRed";
import { RECIPE_IMAGES } from "@/images";
import { formatMarkdown } from "@/utils/textFormatter";
import "./ChatPage.css";

export default function ChatPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const {
    sessionId: existingSessionId,
    existingMessages,
    memberInfo: existingMemberInfo,
    skipToChat,
    fromRegenerate,
    recipe: passedRecipe,
    fromMyPage,
  } = location.state || {};

  const [messages, setMessages] = useState(() => {
    if (existingMessages && existingMessages.length > 0) {
      console.log("[ChatPage] 기존 메시지 복원:", existingMessages);
      return existingMessages;
    }

    if (!skipToChat && !fromRegenerate) {
      localStorage.removeItem("chatMessages");
      localStorage.removeItem("chatMemberInfo");
      console.log("[ChatPage] 새 대화 시작 - localStorage 초기화");
      return [];
    }

    const savedMessages = localStorage.getItem("chatMessages");
    return savedMessages ? JSON.parse(savedMessages) : [];
  });

  const [combinedMemberInfo, setCombinedMemberInfo] = useState(() => {
    if (existingMemberInfo) {
      console.log("[ChatPage] 기존 memberInfo 복원:", existingMemberInfo);
      return existingMemberInfo;
    }

    if (!skipToChat && !fromRegenerate) {
      return null;
    }

    const savedMemberInfo = localStorage.getItem("chatMemberInfo");
    return savedMemberInfo ? JSON.parse(savedMemberInfo) : null;
  });

  const [input, setInput] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  const [flowState, setFlowState] = useState(
    skipToChat ? "FREE_CHAT" : messages.length > 0 ? "FREE_CHAT" : "LOADING",
  );

  const [familyMembers, setFamilyMembers] = useState({});
  const [selectedMembers, setSelectedMembers] = useState([]);
  const [isMemberSelectionLocked, setIsMemberSelectionLocked] = useState(false);
  const [hasRecipeGenerated, setHasRecipeGenerated] = useState(
    messages.length > 0 || skipToChat,
  );

  const wsRef = useRef(null);
  const wsInitializedRef = useRef(false);
  const welcomeMessageSentRef = useRef(false);
  const sessionIdRef = useRef(existingSessionId || crypto.randomUUID());
  const sessionId = sessionIdRef.current;
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const API_URL = import.meta.env.VITE_API_URL || "";
  const WS_URL = import.meta.env.VITE_WS_URL || "";

  // 디버깅용
  useEffect(() => {
    console.log("[ChatPage] 세션 ID:", sessionId);
    console.log("[ChatPage] 재생성 여부:", !!existingSessionId);
    console.log("[ChatPage] skipToChat:", skipToChat);
    console.log("[ChatPage] passedRecipe:", !!passedRecipe);
    console.log("[ChatPage] fromMyPage:", fromMyPage);
    console.log("[ChatPage] 현재 상태:", flowState);
  }, [
    sessionId,
    existingSessionId,
    skipToChat,
    passedRecipe,
    flowState,
    fromMyPage,
  ]);

  // messages 변경시 localStorage 저장
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("chatMessages", JSON.stringify(messages));
    }
  }, [messages]);

  // memberInfo 변경시 localStorage 저장
  useEffect(() => {
    if (combinedMemberInfo) {
      localStorage.setItem(
        "chatMemberInfo",
        JSON.stringify(combinedMemberInfo),
      );
    }
  }, [combinedMemberInfo]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  useEffect(() => {
    if (!passedRecipe) return;

    console.log("[ChatPage] 레시피로 세션 시작");
    console.log("[ChatPage] passedRecipe:", passedRecipe);

    const ingredientsList =
      passedRecipe.ingredients
        ?.map((ing) => `• ${ing.name} ${ing.amount}`)
        .join("\n") || "재료 정보 없음";

    const stepsList =
      passedRecipe.steps
        ?.map((step, idx) => `${idx + 1}. ${step.desc || step}`)
        .join("\n") || "조리법 정보 없음";

    const recipeMessage =
      `[${passedRecipe.title}]\n` +
      `⏱️ ${passedRecipe.cook_time || "30분"} | ` +
      `📊 ${passedRecipe.level || "중급"} | ` +
      `👥 ${passedRecipe.servings || "2인분"}\n\n` +
      `**재료**\n${ingredientsList}\n\n` +
      `**조리법**\n${stepsList}`;

    setMessages([
      {
        role: "system",
        content: `현재 레시피: ${JSON.stringify(passedRecipe)}`,
        timestamp: new Date().toISOString(),
        hidden: true,
      },
      {
        role: "assistant",
        content: recipeMessage,
        timestamp: new Date().toISOString(),
        image: passedRecipe.image,
        hideImage: true,
      },
    ]);
    const memberStr = localStorage.getItem("member");
    const member = memberStr ? JSON.parse(memberStr) : null;
    const passedMemberId = member?.id || 0;

    setCombinedMemberInfo({
      names: ["나"],
      member_id: passedMemberId,
      allergies: [],
      dislikes: [],
      cooking_tools: [],
    });

    setFlowState("FREE_CHAT");
    setHasRecipeGenerated(true);
  }, [passedRecipe]);

  // 에이전트 응답이 끝나면 입력창에 자동 포커스
  useEffect(() => {
    if (!isThinking && flowState === "FREE_CHAT" && isConnected) {
      // 약간의 딜레이를 주어 스크롤이 완료된 후 포커스
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  }, [isThinking, flowState, isConnected]);

  // 가족 선택 또는 "나" 자동 선택
  useEffect(() => {
    if (passedRecipe || skipToChat || fromRegenerate) {
      console.log("[ChatPage] 기존 세션 복원 (skipToChat 또는 재생성)");
      return;
    }

    if (combinedMemberInfo) {
      console.log("[ChatPage] combinedMemberInfo 이미 존재");
      return;
    }

    console.log("[ChatPage] 개인화 정보 로딩 시작...");

    const memberStr = localStorage.getItem("member");
    const member = memberStr ? JSON.parse(memberStr) : null;
    const memberId = member?.id || 0;
    const memberNickname = member?.nickname || "게스트";

    const GUEST_MEMBER_ID = 2;

    const loadFamilyOrPersonalization = async () => {
      try {
        // 게스트 처리
        if (!member || memberId === 0 || memberId === GUEST_MEMBER_ID) {
          const combined = {
            names: ["게스트"],
            member_id: 0,
            allergies: [],
            dislikes: [],
            cooking_tools: [],
          };

          setCombinedMemberInfo(combined);

          setMessages([
            {
              role: "assistant",
              content:
                `안녕하세요, 게스트님! 🥔\n\n` +
                `개인화 정보 없이도 맛있는 레시피를 추천해 드릴게요.\n\n` +
                `로그인하시면 알레르기, 비선호 재료 등을\n맞춤 설정할 수 있어요!\n\n` +
                `지금 바로 요리를 시작해볼까요?`,
              timestamp: new Date().toISOString(),
              showButtons: true,
              buttonType: "start_cooking",
            },
          ]);

          setFlowState("CONFIRM_INFO");
          return;
        }

        // 로그인 사용자: 가족 정보 확인
        const familyRes = await fetch(
          `${API_URL}/api/user/family?member_id=${memberId}`,
        );
        const familyData = await familyRes.json();
        const families = familyData.family_members || [];

        // 가족이 있으면 선택 모드, 없으면 "나"만 자동 선택
        if (families.length > 0) {
          // 가족 선택 모드
          const membersObj = {};
          membersObj[memberNickname] = { type: "member", id: memberId };

          for (const fam of families) {
            const name = fam.relationship || `가족${fam.id}`;
            membersObj[name] = { type: "family", id: fam.id };
          }

          setFamilyMembers(membersObj);

          setMessages([
            {
              role: "assistant",
              content:
                "안녕하세요! 누구를 위한 요리를 만들까요?\n(여러 명 선택 가능)",
              timestamp: new Date().toISOString(),
              showButtons: true,
              buttonType: "select_member",
            },
          ]);

          setFlowState("SELECT_MEMBER");
        } else {
          // "나"만 자동 선택
          const profileRes = await fetch(
            `${API_URL}/api/user/profile?member_id=${memberId}`,
          );
          const profileData = await profileRes.json();

          let memberUtensils = [];
          if (memberId > 0) {
            const utensilRes = await fetch(
              `${API_URL}/api/user/all-constraints?member_id=${memberId}`,
            );
            const utensilData = await utensilRes.json();
            memberUtensils = utensilData.utensils || [];
          }

          const combined = {
            names: ["나"],
            member_id: memberId,
            allergies: profileData.allergies || [],
            dislikes: profileData.dislikes || [],
            cooking_tools: memberUtensils,
          };

          setCombinedMemberInfo(combined);

          let infoLines = [`[ ${memberNickname} ]님을 위한 요리 정보\n`];

          if (combined.allergies.length > 0) {
            infoLines.push(`- 알레르기: ${combined.allergies.join(", ")}\n`);
          }
          if (combined.dislikes.length > 0) {
            infoLines.push(
              `- 싫어하는 음식: ${combined.dislikes.join(", ")}\n`,
            );
          }
          if (combined.cooking_tools.length > 0) {
            infoLines.push(
              `- 사용 가능한 조리도구: ${combined.cooking_tools.join(", ")}\n`,
            );
          }

          const hasPersonalization =
            combined.allergies.length > 0 ||
            combined.dislikes.length > 0 ||
            combined.cooking_tools.length > 0;

          if (!hasPersonalization) {
            infoLines.push(
              `\n아직 등록된 개인화 정보가 없어요.\n마이페이지에서 알레르기, 비선호 음식 등을 등록해보세요!`,
            );
          } else {
            infoLines.push(`\n이 정보가 맞나요?`);
          }

          const infoText = infoLines.join("\n");

          setMessages([
            {
              role: "assistant",
              content: infoText,
              timestamp: new Date().toISOString(),
              showButtons: true,
              buttonType: hasPersonalization ? "confirm_info" : "start_cooking",
            },
          ]);

          setFlowState("CONFIRM_INFO");
        }
      } catch (err) {
        console.error("[ChatPage] 개인화 정보 로딩 실패:", err);
        setCombinedMemberInfo({
          names: ["나"],
          member_id: memberId,
          allergies: [],
          dislikes: [],
          cooking_tools: [],
        });

        setMessages([
          {
            role: "assistant",
            content:
              "개인화 정보를 불러오지 못했어요.\n그래도 요리를 시작할 수 있어요!",
            timestamp: new Date().toISOString(),
            showButtons: true,
            buttonType: "start_cooking",
          },
        ]);

        setFlowState("CONFIRM_INFO");
      }
    };

    loadFamilyOrPersonalization();
  }, [API_URL, skipToChat, fromRegenerate, passedRecipe, combinedMemberInfo]);

  // WebSocket 연결
  useEffect(() => {
    if (flowState !== "FREE_CHAT") {
      console.log("[ChatPage] WebSocket 대기 중... 현재:", flowState);
      return;
    }

    if (wsInitializedRef.current) {
      console.log("[ChatPage] WebSocket 이미 초기화됨, 스킵");
      return;
    }

    if (!combinedMemberInfo) {
      console.log("[ChatPage] combinedMemberInfo 대기 중...");
      return;
    }

    console.log("[ChatPage] WebSocket 연결 시작...");
    wsInitializedRef.current = true;

    const ws = new WebSocket(`${WS_URL}/api/chat/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[WebSocket] Connected");
      setIsConnected(true);

      if (passedRecipe) {
        console.log("[WebSocket] 레시피 포함 컨텍스트 전송");

        const recipeMessage = {
          role: "assistant",
          content: `[${passedRecipe.title}] 레시피입니다.\n재료: ${passedRecipe.ingredients?.map((i) => i.name).join(", ")}\n조리법: ${passedRecipe.steps?.length}단계`,
          image: passedRecipe.image,
        };

        ws.send(
          JSON.stringify({
            type: "init_context",
            member_info: combinedMemberInfo,
            initial_history: [recipeMessage],
          }),
        );
      } else {
        ws.send(
          JSON.stringify({
            type: "init_context",
            member_info: combinedMemberInfo,
          }),
        );
      }

      // 환영 메시지
      if (passedRecipe && !welcomeMessageSentRef.current) {
        welcomeMessageSentRef.current = true;

        setTimeout(() => {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content:
                '이 레시피를 수정하고 싶으신가요?\n예) "덜 맵게 해줘", "재료 바꿔줘", "더 간단하게 만들어줘"',
              timestamp: new Date().toISOString(),
            },
          ]);
        }, 300);
      } else if (
        !passedRecipe &&
        !skipToChat &&
        !welcomeMessageSentRef.current
      ) {
        welcomeMessageSentRef.current = true;

        setTimeout(() => {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content:
                '어떤 요리를 만들고 싶으세요? 자유롭게 말씀해주세요!\n예) "매운 찌개 먹고 싶어요", "간식으로 먹을 요리 알려줘"',
              timestamp: new Date().toISOString(),
            },
          ]);
        }, 300);
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("[WebSocket] Received:", data);

      if (data.type === "session_initialized" && data.db_session_id) {
        console.log("[WebSocket] DB Session ID 수신:", data.db_session_id);
        setDbSessionId(data.db_session_id);
        localStorage.setItem("chatDbSessionId", data.db_session_id);
        return;
      }

      if (data.type === "agent_message") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.content,
            timestamp: new Date().toISOString(),
            image: data.image,
            hideImage: data.hideImage,
          },
        ]);
        setIsThinking(false);
        setHasRecipeGenerated(true);
      } else if (data.type === "not_recipe_related") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.content,
            timestamp: new Date().toISOString(),
            showHomeButton: true,
          },
        ]);
        setIsThinking(false);
        setHasRecipeGenerated(false);
      } else if (data.type === "thinking") {
        setIsThinking(true);
      } else if (data.type === "progress") {
        console.log("[Progress]", data.message);
      } else if (data.type === "error") {
        console.error("Error:", data.message);
        alert(data.message);
        setIsThinking(false);
      }
    };

    ws.onclose = (event) => {
      console.log("[WebSocket] Disconnected", event.code, event.reason);
      setIsConnected(false);
      wsInitializedRef.current = false;
    };

    ws.onerror = (error) => {
      console.error("[WebSocket] Error:", error);
      setIsConnected(false);
    };

    return () => {
      if (
        ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING
      ) {
        console.log("[WebSocket] Cleanup - closing connection");
        ws.close(1000, "Component unmounting");
      }
      wsInitializedRef.current = false;
    };
  }, [
    flowState,
    combinedMemberInfo,
    sessionId,
    WS_URL,
    skipToChat,
    passedRecipe,
  ]);

  // 가족 선택
  const handleSelectMember = (memberName) => {
    setSelectedMembers((prev) =>
      prev.includes(memberName)
        ? prev.filter((name) => name !== memberName)
        : [...prev, memberName],
    );
  };

  // 선택 완료
  const handleConfirmSelection = async () => {
    if (selectedMembers.length === 0) {
      alert("최소 1명을 선택해주세요.");
      return;
    }

    setIsMemberSelectionLocked(true);

    try {
      const memberStr = localStorage.getItem("member");
      const member = memberStr ? JSON.parse(memberStr) : null;
      const memberId = member?.id || 0;

      const allMemberInfo = [];

      for (const name of selectedMembers) {
        const info = familyMembers[name];
        if (!info) continue;

        if (info.type === "member") {
          const res = await fetch(
            `${API_URL}/api/user/profile?member_id=${memberId}`,
          );
          const data = await res.json();
          allMemberInfo.push({
            allergies: data.allergies || [],
            dislikes: data.dislikes || [],
            cooking_tools: [],
          });
        } else {
          const res = await fetch(`${API_URL}/api/user/family/${info.id}`);
          const data = await res.json();
          allMemberInfo.push({
            allergies: data.allergies || [],
            dislikes: data.dislikes || [],
            cooking_tools: [],
          });
        }
      }

      let memberUtensils = [];
      if (memberId > 0) {
        const utensilRes = await fetch(
          `${API_URL}/api/user/all-constraints?member_id=${memberId}`,
        );
        const utensilData = await utensilRes.json();
        memberUtensils = utensilData.utensils || [];
      }

      const combined = {
        names: selectedMembers,
        member_id: memberId,
        allergies: [
          ...new Set(allMemberInfo.flatMap((m) => m.allergies || [])),
        ],
        dislikes: [...new Set(allMemberInfo.flatMap((m) => m.dislikes || []))],
        cooking_tools: memberUtensils,
      };

      setCombinedMemberInfo(combined);

      const namesText = selectedMembers.join(", ");
      let infoLines = [`[ ${namesText} ]님을 위한 요리 정보\n`];

      if (combined.allergies.length > 0) {
        infoLines.push(`- 알레르기: ${combined.allergies.join(", ")}\n`);
      }
      if (combined.dislikes.length > 0) {
        infoLines.push(`- 싫어하는 음식: ${combined.dislikes.join(", ")}\n`);
      }
      if (combined.cooking_tools.length > 0) {
        infoLines.push(
          `- 사용 가능한 조리도구: ${combined.cooking_tools.join(", ")}\n`,
        );
      }

      const hasPersonalization =
        combined.allergies.length > 0 ||
        combined.dislikes.length > 0 ||
        combined.cooking_tools.length > 0;

      if (!hasPersonalization) {
        infoLines.push(
          `\n아직 등록된 개인화 정보가 없어요.\n마이페이지에서 알레르기, 비선호 음식 등을 등록해보세요!`,
        );
      } else {
        infoLines.push(`\n이 정보가 맞나요?`);
      }

      const infoText = infoLines.join("\n");

      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: namesText,
          timestamp: new Date().toISOString(),
        },
        {
          role: "assistant",
          content: infoText,
          timestamp: new Date().toISOString(),
          showButtons: true,
          buttonType: hasPersonalization ? "confirm_info" : "start_cooking",
        },
      ]);

      setFlowState("CONFIRM_INFO");
    } catch (error) {
      console.error("[ChatPage] 멤버 정보 로딩 실패:", error);
      alert("멤버 정보를 불러올 수 없습니다.");
      setIsMemberSelectionLocked(false);
    }
  };

  // 정보 확인
  const handleConfirmInfo = (confirmed, buttonType = "confirm_info") => {
    if (confirmed) {
      const responseMessage =
        buttonType === "start_cooking" ? "좋아요, 시작해볼게요!" : "예, 맞아요";

      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: responseMessage,
          timestamp: new Date().toISOString(),
        },
      ]);

      setFlowState("FREE_CHAT");
      console.log("[ChatPage] 자유 대화 상태로 전환");
    } else {
      console.log("[ChatPage] 마이페이지로 이동");
      navigate({ to: "/mypage" });
    }
  };

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

    const messagePayload = {
      type: "user_message",
      content: input,
      is_recipe_modification: !!passedRecipe,
    };

    console.log("[ChatPage] WebSocket 전송:", messagePayload);
    wsRef.current.send(JSON.stringify(messagePayload));

    setInput("");
    setIsThinking(true);
  };

  const handleGenerateRecipe = () => {
    if (!combinedMemberInfo?.names?.length) {
      alert("가족 정보가 없습니다.");
      return;
    }

    const validMessages = messages.filter(
      (m) => m.role && m.content && typeof m.content === "string",
    );

    console.log("[ChatPage] 레시피 생성 버튼 클릭");

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log("[ChatPage] WebSocket 연결 종료");
      wsRef.current.close(1000, "Navigating to loading");
    }

    navigate({
      to: "/loading",
      state: {
        memberInfo: combinedMemberInfo,
        chatHistory: validMessages,
        sessionId: sessionId,
        isRegeneration: !!fromRegenerate,
      },
    });

    localStorage.setItem(
      "loadingState",
      JSON.stringify({
        memberInfo: combinedMemberInfo,
        chatHistory: validMessages,
        sessionId: sessionId,
        isRegeneration: !!fromRegenerate,
      }),
    );
  };

  // textarea 자동 높이 조절
  const handleTextareaChange = (e) => {
    setInput(e.target.value);

    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "48px";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  };

  return (
    <div
      className="chat-page"
      style={{ backgroundImage: `url(${RECIPE_IMAGES["cook-bg-yellow"]})` }}
    >
      <button className="header-closed" onClick={() => window.history.back()}>
        <img
          src={RECIPE_IMAGES["back-icon"]}
          alt="닫기"
          className="closed-icon"
        />
      </button>
      <div className="chat-header">
        <h1>조리 전, 마지막으로 확인할게요</h1>
      </div>

      <div className="chat-content">
        {flowState === "LOADING" && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <p>로딩 중...</p>
          </div>
        )}

        <div className="messages">
          {messages
            .filter((msg) => !msg.hidden)
            .map((msg, idx) => (
              <div key={idx}>
                <div className={`message ${msg.role}`}>
                  <div
                    className="bubble"
                    dangerouslySetInnerHTML={{
                      __html: formatMarkdown(msg.content),
                    }}
                  />
                </div>
                {msg.image && !msg.hideImage && (
                  <div className="message-image-wrapper">
                    <img
                      src={msg.image}
                      alt="레시피 이미지"
                      className="message-recipe-image"
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                  </div>
                )}
                {msg.showHomeButton && (
                  <div className="home-button-wrapper">
                    <button
                      className="btn-confirm-selection"
                      onClick={() => navigate({ to: "/out-chat" })}
                    >
                      외부 챗봇으로 이동
                    </button>
                  </div>
                )}
                {msg.showButtons && msg.buttonType === "select_member" && (
                  <div className="selection-area">
                    <div className="button-group">
                      {Object.keys(familyMembers).map((name) => (
                        <button
                          key={name}
                          className={`btn-option ${selectedMembers.includes(name) ? "selected" : ""}`}
                          onClick={() => handleSelectMember(name)}
                          disabled={isMemberSelectionLocked}
                        >
                          {name}
                        </button>
                      ))}
                    </div>

                    <button
                      className="btn-confirm-selection"
                      onClick={handleConfirmSelection}
                      disabled={
                        selectedMembers.length === 0 || isMemberSelectionLocked
                      }
                    >
                      {isMemberSelectionLocked ? "선택 완료됨" : "선택 완료"}
                    </button>
                  </div>
                )}
                {msg.showButtons && msg.buttonType === "confirm_info" && (
                  <div className="button-group confirm-group">
                    <button
                      className="btn-option btn-confirm"
                      onClick={() => handleConfirmInfo(true)}
                    >
                      예, 맞아요
                    </button>
                    <button
                      className="btn-option btn-edit"
                      onClick={() => handleConfirmInfo(false)}
                    >
                      수정이 필요해요
                    </button>
                  </div>
                )}
                {msg.showButtons && msg.buttonType === "start_cooking" && (
                  <div className="message assistant">
                    <div className="button-group confirm-group">
                      <button
                        className="btn-option btn-confirm"
                        onClick={() => handleConfirmInfo(true, "start_cooking")}
                      >
                        요리 시작하기
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}

          {isThinking && (
            <div className="message assistant">
              <div className="bubble thinking">
                <div className="thinking-dots">
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

      {flowState === "FREE_CHAT" && (
        <div className="action-area">
          <ButtonRed
            onClick={handleGenerateRecipe}
            disabled={!hasRecipeGenerated || isThinking}
          >
            대화 종료하고 레시피 생성하기
          </ButtonRed>
        </div>
      )}

      {flowState === "FREE_CHAT" && (
        <div className="chat-input-area">
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
            placeholder={isConnected ? "어떤 요리를 원하세요?" : "연결 중..."}
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
      )}
    </div>
  );
}
