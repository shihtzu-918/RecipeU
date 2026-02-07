from fastapi import APIRouter, HTTPException
import requests
import polars as pl
import os

router = APIRouter()

API_KEY = "fce0ae6867c177f191027bfcbc4b1b9e"
CSV_PATH = os.path.join(os.path.dirname(__file__), "korea_lat_lon.csv")

def fetch_weather(lat, lon, sigungu, api_key):
    """
    날씨 정보 가져오기
    Args:
        lat: 위도
        lon: 경도
        sigungu: 현재 위치(시/군/구)
        api_key: OpenWeather API
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "lang": "kr"}
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "sigungu": sigungu,
                "weather_main": data['weather'][0]['main'],
                "weather_desc": data['weather'][0]['description'],
                "temp": data['main']['temp'],
                "humidity": data['main']['humidity'],
                "icon": data['weather'][0]['icon']
            }
    except Exception as e:
        print(f"Error fetching data for {sigungu}: {e}")
    return None

@router.get("/current")
async def get_current_weather(city: str = "서울강남구"):
    """특정 도시 날씨 조회"""
    try:
        df = pl.read_csv(CSV_PATH)
        city_data = df.filter(pl.col("docity") == city)
        
        if city_data.is_empty():
            raise HTTPException(status_code=404, detail=f"{city} 데이터를 찾을 수 없습니다")
        
        row = city_data.row(0, named=True)
        
        weather_data = fetch_weather(
            lat=row["latitude"],
            lon=row["longitude"],
            sigungu=row["docity"],
            api_key=API_KEY
        )
        
        if not weather_data:
            raise HTTPException(status_code=500, detail="날씨 정보를 가져올 수 없습니다")
        
        return weather_data
        
    except Exception as e:
        print(f"[Weather API] 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/location")
async def get_weather_by_location(lat: float, lon: float):
    """위도/경도 기반 날씨 조회"""
    try:
        # 직접 OpenWeather API 호출 (CSV 필요 없음)
        weather_data = fetch_weather(
            lat=lat,
            lon=lon,
            sigungu=f"위치 ({lat:.2f}, {lon:.2f})",
            api_key=API_KEY
        )
        
        if not weather_data:
            raise HTTPException(status_code=500, detail="날씨 정보를 가져올 수 없습니다")
        
        return weather_data
        
    except Exception as e:
        print(f"[Weather API] 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

