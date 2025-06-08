# 1) 스크립트 위치 (auto_run)로 이동
cd "$(dirname "$0")"
# 2) 한 단계 위(/app)로 이동
cd ..

# 3) 로그 디렉터리 생성
mkdir -p logs

# 4) uv run 명령을 실행할 때 환경변수를 한 줄에 함께 지정
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
uv run python src/train_model.py \
  > "logs/train_${TIMESTAMP}.log" 2>&1

# 5) 완료 메시지
echo "[$(date '+%Y-%m-%d %H:%M:%S')] uv run python src/train_model.py 완료 → logs/train_${TIMESTAMP}.log"