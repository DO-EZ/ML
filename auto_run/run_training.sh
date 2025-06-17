#!/usr/bin/env bash
set -e

# 타임스탬프 & 로그 디렉토리
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

# 작업 디렉토리(절대경로) 보장
cd "$(dirname "$0")"/..

# 재학습 시작 (uv run 으로 환경 lock & sync 포함)
echo "▶ TRAIN START: ${TIMESTAMP}"
uv run python -u src/train_model.py \
  --epochs 5 \
  > "${LOG_DIR}/train_${TIMESTAMP}.log" 2>&1
echo "TRAIN COMPLETE: ${TIMESTAMP}"

# 기존 MLflow 서버 종료
pkill -f "mlflow models serve" || true

# 최신 모델 버전 조회
LATEST_VER=$(uv run python - << 'PYCODE'
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="http://mlflow:5000")
vers = client.get_latest_versions("HybridCNN")
print(vers[-1].version)
PYCODE
)

# 모델 서버 재시작
echo "▶ SERVE START version=${LATEST_VER}: ${TIMESTAMP}"
uv run mlflow models serve \
  -m "models:/HybridCNN/${LATEST_VER}" \
  -p 5000 --no-conda --host 0.0.0.0 \
  > "${LOG_DIR}/serve_${TIMESTAMP}.log" 2>&1 &
echo "SERVE COMPLETE version=${LATEST_VER}: ${TIMESTAMP}"
