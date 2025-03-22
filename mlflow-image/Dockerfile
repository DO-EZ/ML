# Python slim 이미지를 기반으로 설정
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MLflow UI 포트 열기
EXPOSE 5000

# 기본 명령어
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]