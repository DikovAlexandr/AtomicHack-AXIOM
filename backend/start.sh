#!/bin/bash
trap "kill 0" SIGINT SIGTERM
export PYTHONPATH=${PYTHONPATH}:/app
echo "Starting FastAPI server..."
uvicorn app:app --host 0.0.0.0 --port 8000 &
echo "Starting Kafka consumer..."
python -u kafka_consumer.py
wait
