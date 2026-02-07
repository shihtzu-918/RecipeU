// src/pages/Home/HomePage.jsx
import { useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import BottomNav from "@/components/BottomNav";
import { RECIPE_IMAGES } from "@/images";
import weatherComments from "@/data/weatherComments.json";
import "./HomePage.css";

export default function HomePage() {
  const navigate = useNavigate();
  const API_URL = import.meta.env.VITE_API_URL || "";

  const [weather, setWeather] = useState({
    temp: "3°",
    desc: "약간흐림",
    icon: RECIPE_IMAGES["main-weather"],
  });

  const [loading, setLoading] = useState(true);
  const [myRecipeCount, setMyRecipeCount] = useState(0);
  const [speechBubble, setSpeechBubble] = useState({
    comment:
      "날이 많이 쌀쌀하네요! \n오늘은 김이 모락모락 나는 국물 요리 어떠세요?",
    recommendation: "",
  });

  const [isBirthday, setIsBirthday] = useState(false);
  const memberStr = localStorage.getItem("member");
  const member = memberStr ? JSON.parse(memberStr) : null;
  const memberId = member?.id || 0;

  // 홈페이지 진입 시 로컬스토리지 정리 (필수 키 제외)
  useEffect(() => {
    const keysToKeep = ["member", "userLocation", "recipesMyCache", "recipesRankingCache"];
    const allKeys = Object.keys(localStorage);
    allKeys.forEach((key) => {
      if (!keysToKeep.includes(key)) {
        localStorage.removeItem(key);
      }
    });
  }, []);

  useEffect(() => {
    const init = async () => {
      loadCachedData();
      const isBirthdayToday = await checkBirthday();
      fetchWeather(isBirthdayToday);
      fetchMyRecipeCount();
    };
    init();
  }, []);

  const loadCachedData = () => {
    try {
      // 날씨 캐시
      const weatherCache = localStorage.getItem("homeWeatherCache");
      if (weatherCache) {
        const {
          weather: cachedWeather,
          speechBubble: cachedBubble,
          timestamp,
        } = JSON.parse(weatherCache);
        const ONE_HOUR = 60 * 60 * 1000;

        // 1시간 이내면 캐시 사용
        if (Date.now() - timestamp < ONE_HOUR) {
          console.log("[HomePage] 날씨 캐시 사용");
          setWeather(cachedWeather);
          setSpeechBubble(cachedBubble);
        }
      }

      // 마이레시피 개수 캐시
      const myRecipeCache = localStorage.getItem("recipesMyCache");
      if (myRecipeCache) {
        const { data, timestamp } = JSON.parse(myRecipeCache);
        const FIVE_MINUTES = 5 * 60 * 1000;

        // 5분 이내면 캐시 사용
        if (Date.now() - timestamp < FIVE_MINUTES) {
          console.log("[HomePage] 마이레시피 개수 캐시 사용");
          setMyRecipeCount((data.recipes || []).length);
        }
      }
    } catch (err) {
      console.error("[HomePage] 캐시 로드 실패:", err);
    }
  };

  // 날씨 데이터 캐싱 함수
  const cacheWeatherData = (weatherData, bubbleData) => {
    try {
      const cache = {
        weather: weatherData,
        speechBubble: bubbleData,
        timestamp: Date.now(),
      };
      localStorage.setItem("homeWeatherCache", JSON.stringify(cache));
      console.log("[HomePage] 날씨 데이터 캐싱 완료");
    } catch (err) {
      console.error("[HomePage] 날씨 캐싱 실패:", err);
    }
  };

  useEffect(() => {
    const prefetchRecipes = async () => {
      try {
        console.log("[HomePage] 레시피 데이터 prefetch 시작");

        // 1. 랭킹 레시피 prefetch
        const rankingResponse = await fetch(
          `${API_URL}/api/rankings/today?limit=100`,
        );
        const rankingData = await rankingResponse.json();

        // 캐싱 (5분 유효)
        const rankingCache = {
          data: rankingData,
          timestamp: Date.now(),
        };
        localStorage.setItem(
          "recipesRankingCache",
          JSON.stringify(rankingCache),
        );

        // 이미지 prefetch (상위 20개만)
        rankingData.recipes?.slice(0, 20).forEach((recipe) => {
          if (recipe.image) {
            const img = new Image();
            img.src = recipe.image;
          }
        });

        console.log(
          "[HomePage] 랭킹 레시피 prefetch 완료:",
          rankingData.recipes?.length,
        );

        // 2. 마이 레시피 prefetch (로그인한 경우만)
        if (memberId !== 0) {
          const myRecipeResponse = await fetch(
            `${API_URL}/api/recipe/list?member_id=${memberId}`,
          );
          const myRecipeData = await myRecipeResponse.json();

          const myRecipeCache = {
            data: myRecipeData,
            timestamp: Date.now(),
          };
          localStorage.setItem("recipesMyCache", JSON.stringify(myRecipeCache));

          // 이미지 prefetch (상위 10개만)
          myRecipeData.recipes?.slice(0, 10).forEach((recipe) => {
            if (recipe.image) {
              const img = new Image();
              img.src = recipe.image;
            }
          });

          console.log(
            "[HomePage] 마이 레시피 prefetch 완료:",
            myRecipeData.recipes?.length,
          );
        }
      } catch (err) {
        console.log("[HomePage] prefetch 실패 (무시):", err);
      }
    };

    // 페이지 로드 후 2초 뒤 실행 (홈 로딩에 방해 안 되게)
    const timer = setTimeout(prefetchRecipes, 2000);

    return () => clearTimeout(timer);
  }, [memberId, API_URL]);

  const getWeatherIcon = (weatherDesc) => {
    const desc = weatherDesc.toLowerCase();

    if (
      desc.includes("맑") ||
      desc.includes("clear") ||
      desc.includes("sunny")
    ) {
      return RECIPE_IMAGES["sun"];
    } else if (desc.includes("비") || desc.includes("rain")) {
      return RECIPE_IMAGES["rain"];
    } else if (desc.includes("눈") || desc.includes("snow")) {
      return RECIPE_IMAGES["snow"];
    } else if (
      desc.includes("번개") ||
      desc.includes("thunder") ||
      desc.includes("storm")
    ) {
      return RECIPE_IMAGES["storm"];
    } else if (desc.includes("바람") || desc.includes("wind")) {
      return RECIPE_IMAGES["wind"];
    } else if (
      desc.includes("흐") ||
      desc.includes("구름") ||
      desc.includes("cloud")
    ) {
      return RECIPE_IMAGES["cloud"];
    } else {
      return RECIPE_IMAGES["main-weather"];
    }
  };

  const getShortWeatherDesc = (weatherDesc) => {
    const desc = weatherDesc.toLowerCase();

    if (desc.includes("맑")) return "맑음";
    if (desc.includes("비")) return "비";
    if (desc.includes("눈")) return "눈";
    if (desc.includes("번개") || desc.includes("storm")) return "천둥번개";
    if (desc.includes("바람")) return "바람";
    if (desc.includes("흐") || desc.includes("구름")) return "흐림";

    return weatherDesc;
  };

  const getTimeOfDay = () => {
    const hour = new Date().getHours();
    if (hour >= 6 && hour < 12) return "아침";
    if (hour >= 12 && hour < 18) return "점심";
    if (hour >= 18 && hour < 24) return "저녁";
    return "새벽";
  };

  const getTempCategory = (temp) => {
    if (temp < 10) return "추움";
    if (temp > 25) return "더움";
    return "쾌적";
  };

  const getWeatherCategory = (weatherDesc) => {
    const desc = weatherDesc.toLowerCase();
    if (desc.includes("맑") || desc.includes("clear")) return "맑음";
    if (desc.includes("비") || desc.includes("rain")) return "비옴";
    if (desc.includes("눈") || desc.includes("snow")) return "눈옴";
    return "구름 낌";
  };

  const getWeatherComment = (temp, weatherDesc) => {
    const timeOfDay = getTimeOfDay();
    const tempCategory = getTempCategory(temp);
    const weatherCategory = getWeatherCategory(weatherDesc);

    const matchedComment = weatherComments.find(
      (item) =>
        item.time === timeOfDay &&
        item.temp === tempCategory &&
        item.weather === weatherCategory,
    );

    if (matchedComment) {
      return {
        comment: matchedComment.comment,
        recommendation: matchedComment.recommendation,
      };
    }

    return {
      comment:
        "날이 많이 쌀쌀하네요! \n오늘은 김이 모락모락 나는 국물 요리 어떠세요?",
      recommendation: "",
    };
  };

  // 위치 정보 캐싱 함수
  const getCachedLocation = () => {
    const cached = localStorage.getItem("userLocation");
    if (cached) {
      try {
        const { lat, lon, timestamp } = JSON.parse(cached);
        const now = Date.now();
        const ONE_DAY = 24 * 60 * 60 * 1000; // 24시간

        // 24시간 이내면 캐시 사용
        if (now - timestamp < ONE_DAY) {
          console.log("[Weather] 캐시된 위치 사용:", { lat, lon });
          return { lat, lon };
        }
      } catch (err) {
        console.error("[Weather] 캐시 파싱 실패:", err);
      }
    }
    return null;
  };

  // 위치 정보 저장 함수
  const setCachedLocation = (lat, lon) => {
    const locationData = {
      lat,
      lon,
      timestamp: Date.now(),
    };
    localStorage.setItem("userLocation", JSON.stringify(locationData));
    console.log("[Weather] 위치 정보 캐시 저장:", locationData);
  };

  const fetchWeather = async (isBirthdayToday = false) => {
    try {
      // 먼저 캐시된 위치 확인
      const cachedLocation = getCachedLocation();

      if (cachedLocation) {
        // 캐시된 위치로 날씨 가져오기
        await fetchWeatherByLocation(
          cachedLocation.lat,
          cachedLocation.lon,
          isBirthdayToday,
        );
        return;
      }

      // 캐시 없으면 새로 위치 가져오기
      if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
          async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            console.log("[Weather] 새 위치 정보 획득:", { lat, lon });

            // 위치 정보 캐싱
            setCachedLocation(lat, lon);

            // 날씨 가져오기
            await fetchWeatherByLocation(lat, lon, isBirthdayToday);
          },
          () => {
            console.log("[Weather] 위치 권한 거부 - 기본 위치 사용");
            fetchWeatherByCity("서울강남구", isBirthdayToday);
          },
          {
            enableHighAccuracy: false,
            timeout: 5000,
            maximumAge: 300000,
          },
        );
      } else {
        console.log("[Weather] Geolocation 미지원 - 기본 위치 사용");
        fetchWeatherByCity("서울강남구", isBirthdayToday);
      }
    } catch (error) {
      console.error("[Weather] 에러:", error);
      setLoading(false);
    }
  };

  // fetchWeatherByLocation 수정
  const fetchWeatherByLocation = async (lat, lon, isBirthdayToday = false) => {
    try {
      const response = await fetch(
        `${API_URL}/api/weather/location?lat=${lat}&lon=${lon}`,
      );

      if (response.ok) {
        const data = await response.json();

        const bubbleContent = getWeatherComment(data.temp, data.weather_desc);

        if (!isBirthdayToday) {
          setSpeechBubble(bubbleContent);
        }

        const weatherData = {
          temp: `${Math.round(data.temp)}°`,
          desc: getShortWeatherDesc(data.weather_desc),
          icon: getWeatherIcon(data.weather_desc),
        };

        setWeather(weatherData);

        // 날씨 데이터 캐싱
        if (!isBirthdayToday) {
          cacheWeatherData(weatherData, bubbleContent);
        }
      } else {
        fetchWeatherByCity("서울강남구", isBirthdayToday);
      }
    } catch (error) {
      console.error("[Weather] API 에러:", error);
      fetchWeatherByCity("서울강남구", isBirthdayToday);
    } finally {
      setLoading(false);
    }
  };

  const fetchMyRecipeCount = async () => {
    try {
      const response = await fetch(
        `${API_URL}/api/recipe/list?member_id=${memberId}`,
      );
      if (!response.ok) return;

      const data = await response.json();
      setMyRecipeCount((data.recipes || []).length);

      // 마이레시피 캐싱
      const cache = {
        data,
        timestamp: Date.now(),
      };
      localStorage.setItem("recipesMyCache", JSON.stringify(cache));
      console.log("[HomePage] 마이레시피 캐싱 완료");
    } catch (error) {
      console.error("마이 레시피 개수 불러오기 실패:", error);
    }
  };

  const checkBirthday = async () => {
    try {
      const FORCE_BIRTHDAY_TEST = false;

      if (FORCE_BIRTHDAY_TEST) {
        setIsBirthday(true);
        setSpeechBubble({
          comment:
            "일 년 중 가장 소중한 생일을 진심으로 축하드려요!\n행복을 기원하며 따뜻한 미역국 한 그릇은 어떨까요?",
          recommendation: "",
        });
        return true;
      }

      if (memberId === 0) {
        return false;
      }

      const response = await fetch(
        `${API_URL}/api/user/profile?member_id=${memberId}`,
      );

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      const birthday = data.birthday;

      if (!birthday) {
        return false;
      }

      console.log(birthday);

      const today = new Date();
      const todayMonth = String(today.getMonth() + 1).padStart(2, "0");
      const todayDay = String(today.getDate()).padStart(2, "0");
      const todayStr = `${todayMonth}-${todayDay}`;

      if (birthday === todayStr || birthday === `${todayMonth}${todayDay}`) {
        setIsBirthday(true);
        setSpeechBubble({
          comment:
            "일 년 중 가장 소중한 생일을 진심으로 축하드려요!\n행복을 기원하며 따뜻한 미역국 한 그릇은 어떨까요?",
          recommendation: "",
        });
        return true;
      } else {
        setIsBirthday(false);
        return false;
      }
    } catch (error) {
      console.error("생일 확인 실패:", error);
      return false;
    }
  };

  const fetchWeatherByCity = async (city, isBirthdayToday = false) => {
    try {
      const response = await fetch(
        `${API_URL}/api/weather/current?city=${city}`,
      );

      if (response.ok) {
        const data = await response.json();

        const bubbleContent = getWeatherComment(data.temp, data.weather_desc);

        if (!isBirthdayToday) {
          setSpeechBubble(bubbleContent);
        }

        const weatherData = {
          temp: `${Math.round(data.temp)}°`,
          desc: getShortWeatherDesc(data.weather_desc),
          icon: getWeatherIcon(data.weather_desc),
        };

        setWeather(weatherData);

        if (!isBirthdayToday) {
          cacheWeatherData(weatherData, bubbleContent);
        }
      }
    } catch (error) {
      console.error("폴백 날씨 API 에러:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-container">
      <div
        className="home-bg"
        style={{ backgroundImage: `url(${RECIPE_IMAGES["main-bg"]})` }}
        fetchpriority="high"
        loading="eager"
      />

      <div className="weather-box">
        <img src={weather.icon} alt="날씨" />
        <span className="weather-temp">{weather.temp}</span>
        <span className="weather-desc">{weather.desc}</span>
      </div>

      <div className="mypage-box">
        <span className="mypage-desc">마이페이지</span>
      </div>

      <img
        src={RECIPE_IMAGES["potato-face"]}
        alt="프로필"
        className="profile-icon"
        onClick={() => navigate({ to: "/mypage" })}
        style={{ cursor: "pointer" }}
      />

      <div className="speech-bubble">
        {speechBubble.comment.split("\n").map((line, i) => (
          <span key={i}>
            {line}
            {i < speechBubble.comment.split("\n").length - 1 && <br />}
          </span>
        ))}
      </div>

      <img
        src={
          isBirthday
            ? RECIPE_IMAGES["birthday-main-character_v2"]
            : RECIPE_IMAGES["main-character"]
        }
        alt="캐릭터"
        className="main-character"
        fetchpriority="high"
        loading="eager"
      />

      <div className="home-scroll">
        <div
          className="card card-large"
          onClick={() => navigate({ to: "/cook-start" })}
          style={{ cursor: "pointer" }}
        >
          <img
            src={RECIPE_IMAGES["main-profile"]}
            className="card-icon"
            alt="감자"
          />

          <div className="card-text">
            <h3>오늘의 맞춤 레시피는?</h3>
            <p>
              오늘 뭐 먹어야할지 모르겠다면?
              <br />
              지금 바로 맞춤 레시피를 알아보세요!
            </p>
          </div>

          <img
            src={RECIPE_IMAGES["main-next"]}
            className="card-action"
            alt="이동"
          />
        </div>

        <div
          className="card card-report"
          onClick={() =>
            window.open(
              "https://form.naver.com/response/iaXgaNcPxVW-UiydhoyZoA",
              "_blank",
              "noopener,noreferrer",
            )
          }
        >
          <img
            src={RECIPE_IMAGES["main-profile"]}
            className="card-icon"
            alt="감자"
          />
          <div className="card-text">
            <h3 className="card-h3">사용하다가 불편한 점이 있으셨나요?</h3>
            <p>리포트를 보내주시면 레시퓨의 발전에 큰 도움이 됩니다!</p>
          </div>
        </div>

        <div className="card-row">
          <div
            className="card small"
            onClick={() => navigate({ to: "/recipes/my" })}
            style={{ cursor: "pointer" }}
          >
            <div className="card-small-top">
              <img
                src={RECIPE_IMAGES["main-profile"]}
                className="card-icon"
                alt="감자"
              />
              <span className="highlight">{myRecipeCount}개</span>
            </div>

            <h3>마이 레시피</h3>
            <p>
              이전에 함께 만들었던,
              <br />
              레시피를 확인해요!
            </p>
          </div>

          <div
            className="card small"
            onClick={() => navigate({ to: "/recipes" })}
            style={{ cursor: "pointer" }}
          >
            <div className="card-small-top">
              <img
                src={RECIPE_IMAGES["main-profile"]}
                className="card-icon"
                alt="감자"
              />
            </div>

            <h3>TOP 100 레시피</h3>
            <p>
              인기 있는 다른 레시피가
              <br />
              궁금하다면?
            </p>
          </div>
        </div>
      </div>

      <BottomNav />
    </div>
  );
}
