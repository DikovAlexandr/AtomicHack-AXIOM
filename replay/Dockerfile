FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY log_replay.py .

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python3 -c 'from kafka.admin import KafkaAdminClient; admin = KafkaAdminClient(bootstrap_servers="kafka:9092", request_timeout_ms=5000); topics = admin.list_topics(timeout_ms=5000); exit(0 if topics else 1)'

ENV KAFKA_BOOTSTRAP=kafka:9092 \
    TOPIC_PREFIX=logs \
    LOG_DIR=/logs \
    SPEED=10 \
    JITTER_MS=0

CMD ["python3", "log_replay.py"]
