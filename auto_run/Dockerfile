FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip cron bash jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="/app/src"

COPY pyproject.toml uv.lock ./
RUN pip3 install --upgrade pip && \
    pip3 install uv && \
    uv sync

COPY src/           /app/src/
COPY auto_run/      /app/auto_run/
RUN sed -i 's/\r$//' /app/auto_run/run_training.sh
COPY main.py        /app/
COPY auto_run/crontab.txt /etc/cron.d/ml_training

RUN chmod +x /app/auto_run/run_training.sh && \
    chmod 644 /etc/cron.d/ml_training && \
    crontab /etc/cron.d/ml_training && \
    mkdir -p /app/logs

CMD ["cron", "-f"]