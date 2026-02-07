FROM python:3.11-slim

WORKDIR /app

# Git 설치 (MeloTTS 설치를 위해 필요)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 노출
EXPOSE 8080

# 실행
CMD ["python", "run.py"]
