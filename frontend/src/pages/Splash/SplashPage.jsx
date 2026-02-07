import { useNavigate } from "@tanstack/react-router";
import { RECIPE_IMAGES } from "@/images";
import "./SplashPage.css";

const API_URL = import.meta.env.VITE_API_URL || "";

export default function SplashPage() {
    const navigate = useNavigate();

    const goHome = async () => {
        try {
            // DB에서 게스트 정보 가져오기
            const res = await fetch(`${API_URL}/api/user/profile?member_id=2`);
            if (!res.ok) throw new Error("게스트 정보 로드 실패");

            const data = await res.json();

            const guestUser = {
                id: data.id,
                nickname: data.name,
                email: data.email,
                name: data.name,
                birthday: data.birthday,
                mem_photo: "https://i.imgur.com/OisBNf2.jpeg",
                profile_image: null,
            };

            localStorage.setItem("member", JSON.stringify(guestUser));
            navigate({ to: "/home" });
        } catch (err) {
            console.error("❌ 게스트 정보 로드 실패:", err);
            console.warn("⚠️ 폴백 데이터 사용 중 - DB 연동을 확인하세요!");

            // 폴백: 기본 게스트 정보 사용
            const guestUser = {
                id: 2,
                nickname: "게스트",
                email: "guest@recipeu.com",
                name: "게스트",
                birthday: "01-01",
                mem_photo: "https://i.imgur.com/OisBNf2.jpeg",
                profile_image: null,
            };
            localStorage.setItem("member", JSON.stringify(guestUser));
            navigate({ to: "/home" });
        }
    };

    const goPeuExperience = async () => {
        try {
            // DB에서 퓨 정보 가져오기
            const res = await fetch(`${API_URL}/api/user/profile?member_id=1`);
            if (!res.ok) throw new Error("퓨 정보 로드 실패");

            const data = await res.json();

            const peuUser = {
                id: data.id,
                nickname: data.name,
                email: data.email,
                name: data.name,
                birthday: data.birthday,
                mem_photo: "https://i.imgur.com/OisBNf2.jpeg",
                profile_image: null,
            };

            localStorage.setItem("member", JSON.stringify(peuUser));
            navigate({ to: "/home" });
        } catch (err) {
            console.error("❌ 퓨 정보 로드 실패:", err);
            console.warn("⚠️ 폴백 데이터 사용 중 - DB 연동을 확인하세요!");

            // 폴백: 기본 퓨 정보 사용
            const peuUser = {
                id: 1,
                nickname: "퓨",
                email: "peu@recipeu.com",
                name: "퓨",
                birthday: "02-06",
                mem_photo: "https://i.imgur.com/OisBNf2.jpeg",
                profile_image: null,
            };
            localStorage.setItem("member", JSON.stringify(peuUser));
            navigate({ to: "/home" });
        }
    };

    const goNaverLogin = async () => {
        try {
            const callbackUrl = `${window.location.origin}/naver-callback`;
            const res = await fetch(
                `${API_URL}/api/auth/login-url?callback_url=${encodeURIComponent(callbackUrl)}`,
            );
            const data = await res.json();

            if (data.url) {
                sessionStorage.setItem("naver_oauth_state", data.state);
                window.location.href = data.url;
            }
        } catch (err) {
            console.error("네이버 로그인 URL 요청 실패:", err);
        }
    };

    return (
        <div
            className="splash-container"
            style={{ backgroundImage: `url(${RECIPE_IMAGES["splash-bg"]})` }}
        >
            {/* RecipeU */}
            <p className="splash-recipeu">RecipeU</p>

            {/* 레시퓨 */}
            <div className="splash-title-row">
                <span className="splash-title-char splash-title-char--reo">레</span>
                <span className="splash-title-char splash-title-char--si">시</span>
                <span className="splash-title-char splash-title-char--peu">퓨</span>
            </div>

            {/* 캐릭터 이미지 */}
            <img
                src={RECIPE_IMAGES["splash-potato"]}
                alt="레시퓨 캐릭터"
                className="splash-character-img"
            />

            {/* 네이버 로그인 */}
            <button className="splash-naver-btn" onClick={goNaverLogin}>
                <img
                    src={RECIPE_IMAGES["login-naver"]}
                    alt="네이버 로그인"
                    className="splash-naver-btn-img"
                />
            </button>
            {/* 퓨로 체험해보기 */}
            <button className="splash-peu-btn" onClick={goPeuExperience}>
                🥔 퓨로 체험해보기
            </button>
            {/* 로그인 없이 사용해보기 */}
            {/* <button className="splash-guest-btn" onClick={goHome}>
        로그인 없이 사용해보기
      </button>       */}
        </div>
    );
}
