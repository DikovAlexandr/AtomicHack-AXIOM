"""
Kafka consumer for real-time log processing and correlation analysis.
"""
import sys
print("--- KAFKA CONSUMER SCRIPT STARTED ---", file=sys.stderr)
import json
import os
import logging
import asyncio
from collections import deque
from datetime import datetime

import pandas as pd
import torch
from dateutil import parser as dtparser
from kafka import KafkaConsumer
from sentence_transformers import SentenceTransformer

from processing import (
    preprocess_message, find_best_match_id_by_embedding,
    CORRELATION_WINDOW, SIMILARITY_THRESHOLD, MODEL_NAME, DEVICE
)
from telegram_bot import bot_manager, send_notification
from realtime_monitor import get_realtime_monitor

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9292")
KAFKA_TOPIC_PREFIX = os.getenv("TOPIC_PREFIX", "logs")
ANOMALIES_CSV_PATH = os.getenv("ANOMALIES_CSV_PATH", "data/anomalies_problems.csv")

logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("--- KAFKA CONSUMER LOGGER INITIALIZED ---")
logger.info(f"Connecting to KAFKA_BOOTSTRAP: {KAFKA_BOOTSTRAP}")


class LogCorrelationConsumer:
    """
    Consumes log messages from Kafka, processes them to find correlations
    between errors and warnings, and integrates with a real-time monitor.
    """

    def __init__(self):
        """Initializes the consumer, ML model, and Kafka connection."""
        logger.info("Initializing consumer...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.anomalies_problems_df = pd.read_csv(ANOMALIES_CSV_PATH, sep=';')
        self.anomalies_problems_df.rename(columns={
            'ID проблемы': 'problem_id',
            'Проблема': 'problem_text',
            'ID аномалии': 'anomaly_id',
            'Аномалия': 'anomaly_text'
        }, inplace=True)

        self._prepare_reference_embeddings()

        self.recent_errors = deque()
        self.recent_warnings = deque()

        self.consumer = KafkaConsumer(
            bootstrap_servers=[KAFKA_BOOTSTRAP],
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            reconnect_backoff_ms=2000,
            retry_backoff_ms=1000,
            # Add a group_id for consumer group behavior
            group_id='log_analyzer_group',
            # Add a client_id for easier identification in logs
            client_id=f'log-analyzer-consumer-{os.getpid()}'
        )
        self.consumer.subscribe(pattern=f'^{KAFKA_TOPIC_PREFIX}.*')
        logger.info(f"Consumer initialized and subscribed to topics matching '{KAFKA_TOPIC_PREFIX}.*'")

    def _prepare_reference_embeddings(self):
        """Pre-computes embeddings for known problems and anomalies."""
        self.problems_df = self.anomalies_problems_df[['problem_id', 'problem_text']].drop_duplicates().rename(
            columns={'problem_id': 'id', 'problem_text': 'text'})
        self.anomalies_df = self.anomalies_problems_df[['anomaly_id', 'anomaly_text']].drop_duplicates().rename(
            columns={'anomaly_id': 'id', 'anomaly_text': 'text'})

        with torch.inference_mode():
            cleaned_problems = [preprocess_message(t) for t in self.problems_df['text']]
            self.problems_df['embedding'] = list(self.model.encode(cleaned_problems, convert_to_tensor=True, device=DEVICE))

            cleaned_anomalies = [preprocess_message(t) for t in self.anomalies_df['text']]
            self.anomalies_df['embedding'] = list(self.model.encode(cleaned_anomalies, convert_to_tensor=True, device=DEVICE))
        logger.info("Reference embeddings are ready.")

    def _prune_old_events(self):
        """Removes events from deques that are older than the correlation window."""
        now = datetime.now()
        while self.recent_errors and (now - self.recent_errors[0]['timestamp']) > CORRELATION_WINDOW:
            self.recent_errors.popleft()
        while self.recent_warnings and (now - self.recent_warnings[0]['timestamp']) > CORRELATION_WINDOW:
            self.recent_warnings.popleft()

    def process_message(self, msg_data: dict):
        """
        Processes a single message from Kafka.
        This includes real-time forwarding and correlation analysis.
        """
        self._handle_realtime_processing(msg_data)

        log_level = msg_data.get('level')
        if log_level not in ["ERROR", "WARNING"]:
            return

        event = {
            "timestamp": dtparser.parse(msg_data['timestamp']).replace(tzinfo=None),
            "type": log_level,
            "message": msg_data['message'],
            "full_line": msg_data['raw']
        }

        cleaned_msg = preprocess_message(event['message'])
        with torch.inference_mode():
            event['embedding'] = self.model.encode(cleaned_msg, convert_to_tensor=True, device=DEVICE)

        self._find_and_process_correlations(event)

    def _handle_realtime_processing(self, msg_data: dict):
        """Sends events to the real-time monitor and triggers real-time correlation."""
        realtime_monitor = get_realtime_monitor()
        if not realtime_monitor:
            return

        try:
            realtime_event = asyncio.run(realtime_monitor.process_realtime_log(msg_data))

            if realtime_event:
                asyncio.run(realtime_monitor.broadcast_json({"type": "event", "data": realtime_event.to_dict()}))
                asyncio.run(self._process_realtime_correlations(realtime_event, realtime_monitor))
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")

    def _find_and_process_correlations(self, event: dict):
        """Finds correlations based on embeddings and sends notifications."""
        source_deque = self.recent_errors if event['type'] == 'WARNING' else self.recent_warnings
        target_deque = self.recent_warnings if event['type'] == 'WARNING' else self.recent_errors

        for old_event in list(source_deque):
            if abs(event['timestamp'] - old_event['timestamp']) > CORRELATION_WINDOW:
                continue

            similarity = torch.nn.functional.cosine_similarity(event['embedding'], old_event['embedding'], dim=0).item()
            if similarity < SIMILARITY_THRESHOLD:
                continue

            error_event = event if event['type'] == 'ERROR' else old_event
            warning_event = event if event['type'] == 'WARNING' else old_event

            problem_id, _ = find_best_match_id_by_embedding(error_event['embedding'], self.problems_df)
            anomaly_id, _ = find_best_match_id_by_embedding(warning_event['embedding'], self.anomalies_df)

            if problem_id != -1 and anomaly_id != -1:
                is_valid = not self.anomalies_problems_df[
                    (self.anomalies_problems_df['problem_id'] == problem_id) &
                    (self.anomalies_problems_df['anomaly_id'] == anomaly_id)
                ].empty

                if is_valid:
                    result = {
                        "anomaly_id": anomaly_id,
                        "problem_id": problem_id,
                        "log": error_event['full_line'],
                        "correlation_timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"CORRELATION FOUND: {result}")
                    send_notification(result)

        target_deque.append(event)
        self._prune_old_events()

    async def _process_realtime_correlations(self, realtime_event, realtime_monitor):
        """Processes and broadcasts correlations in real-time."""
        try:
            correlations = await realtime_monitor.find_correlations_realtime(
                realtime_event, self.problems_df, self.anomalies_df
            )
            for correlation in correlations:
                await realtime_monitor.broadcast_json({"type": "correlation", "data": correlation})
        except Exception as e:
            logger.error(f"Error processing real-time correlations: {e}")

    def run(self):
        """Starts the Kafka consumer loop."""
        logger.info("Listening for messages...")
        try:
            for message in self.consumer:
                logger.info(f"Received message: topic='{message.topic}', partition={message.partition}, offset={message.offset}")
                logger.debug(f"Message value: {message.value}")
                self.process_message(message.value)
        except Exception as e:
            logger.critical(f"Critical error in Kafka consumer loop: {e}", exc_info=True)
        finally:
            logger.info("Closing Kafka consumer.")
            self.consumer.close()


if __name__ == "__main__":
    if bot_manager:
        bot_manager.start_bot_in_thread()

    consumer = LogCorrelationConsumer()
    consumer.run()