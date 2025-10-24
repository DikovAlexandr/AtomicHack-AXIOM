"""
Real-time monitoring module.
Includes WebSocket support, system metrics, and real-time notifications.
"""
import asyncio
import json
import logging
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Set, Optional, Any

import psutil
import torch
import pandas as pd
from dateutil import parser as dtparser
from fastapi import WebSocket
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from processing import (
    preprocess_message, parse_log_line,
    CORRELATION_WINDOW, SIMILARITY_THRESHOLD, DEVICE,
    get_redis_connection, find_best_match_id_by_embedding
)
from telegram_bot import send_notification

logger = logging.getLogger(__name__)


@dataclass
class RealtimeEvent:
    """Structure for a real-time event."""
    timestamp: datetime
    event_type: str  # 'ERROR', 'WARNING', 'INFO'
    message: str
    component: str
    file_name: str
    line_number: int
    embedding: Optional[torch.Tensor] = None

    def to_dict(self):
        d = asdict(self)
        d.pop('embedding', None)  # Don't send embedding to frontend
        return d


@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int
    kafka_messages_processed: int
    correlations_found: int


class RealtimeMonitor:
    """Main class for real-time monitoring."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.redis_conn = get_redis_connection()
        self.active_connections: Set[WebSocket] = set()

        self.recent_errors = deque(maxlen=500)
        self.recent_warnings = deque(maxlen=500)
        self.recent_infos = deque(maxlen=500)
        self.recent_correlations = deque(maxlen=100)
        self.metrics_history = deque(maxlen=100)

        self.kafka_messages_count = 0
        self.correlations_count = 0
        self.event_stats = defaultdict(int)

        self._background_tasks = set()
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Starts background tasks for metric collection using asyncio."""
        metrics_task = asyncio.create_task(self._collect_metrics_periodically())
        self._background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self._background_tasks.discard)
        logger.info("Background metrics collection started.")

    async def _collect_metrics_periodically(self):
        """Periodically collects and broadcasts system metrics."""
        while True:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                await self.broadcast_json({"type": "metrics", "data": asdict(metrics)})
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            await asyncio.sleep(10)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collects system metrics in a non-blocking way."""
        loop = asyncio.get_running_loop()
        
        # Run synchronous psutil calls in a thread pool
        cpu_percent = await loop.run_in_executor(None, psutil.cpu_percent, 1)
        memory = await loop.run_in_executor(None, psutil.virtual_memory)
        disk = await loop.run_in_executor(None, psutil.disk_usage, '/')
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            active_connections=len(self.active_connections),
            kafka_messages_processed=self.kafka_messages_count,
            correlations_found=self.correlations_count
        )

    async def connect_websocket(self, websocket: WebSocket):
        """Accepts a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    async def disconnect_websocket(self, websocket: WebSocket):
        """Handles a WebSocket disconnection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        """Broadcasts a JSON message to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        message = json.dumps(data, default=str)
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.add(websocket)
        
        for websocket in disconnected:
            await self.disconnect_websocket(websocket)

    async def process_realtime_log(self, log_data: Dict[str, Any]) -> Optional[RealtimeEvent]:
        """Parses and processes a single log entry in real-time."""
        try:
            if not all(k in log_data for k in ['timestamp', 'level', 'message', 'component']):
                logger.warning(f"Skipping malformed log data: {log_data}")
                return None

            cleaned_message = preprocess_message(log_data['message'])

            # Run blocking model inference in a thread pool
            embedding = await asyncio.to_thread(
                self.model.encode, cleaned_message, convert_to_tensor=True, device=DEVICE
            )

            realtime_event = RealtimeEvent(
                timestamp=dtparser.parse(log_data['timestamp']),
                event_type=log_data['level'],
                message=log_data['message'],
                component=log_data['component'],
                file_name=log_data.get('source', 'unknown'),
                line_number=log_data.get('line_number', 0),
                embedding=embedding
            )

            if realtime_event.event_type == 'ERROR':
                self.recent_errors.append(realtime_event)
            elif realtime_event.event_type == 'WARNING':
                self.recent_warnings.append(realtime_event)
            elif realtime_event.event_type == 'INFO':
                self.recent_infos.append(realtime_event)

            self.event_stats[realtime_event.event_type] += 1
            self.kafka_messages_count += 1

            return realtime_event
        except Exception as e:
            logger.error(f"Error processing real-time log: {e}", exc_info=True)
            return None

    async def find_correlations_for_event(self, event: RealtimeEvent, problems_df: pd.DataFrame, anomalies_df: pd.DataFrame):
        """Finds correlations for a new event against a buffer of recent events."""
        if event.event_type not in ['ERROR', 'WARNING']:
            return

        source_deque = self.recent_warnings if event.event_type == 'ERROR' else self.recent_errors
        
        for old_event in list(source_deque):
            if abs(event.timestamp - old_event.timestamp) > CORRELATION_WINDOW:
                continue

            # This part is synchronous but fast (tensor operation)
            similarity = util.cos_sim(event.embedding, old_event.embedding).item()

            if similarity >= SIMILARITY_THRESHOLD:
                error_event = event if event.event_type == 'ERROR' else old_event
                warning_event = event if event.event_type == 'WARNING' else old_event

                # These are also fast tensor operations
                problem_id, _ = find_best_match_id_by_embedding(error_event.embedding, problems_df)
                anomaly_id, _ = find_best_match_id_by_embedding(warning_event.embedding, anomalies_df)

                if problem_id != -1 and anomaly_id != -1:
                    correlation = {
                        "problem_id": problem_id, "anomaly_id": anomaly_id,
                        "problem_text": problems_df.loc[problems_df['id'] == problem_id, 'text'].iloc[0],
                        "anomaly_text": anomalies_df.loc[anomalies_df['id'] == anomaly_id, 'text'].iloc[0],
                        "log": error_event.message, "similarity": similarity,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.recent_correlations.append(correlation)
                    self.correlations_count += 1
                    
                    await self.broadcast_json({"type": "correlation", "data": correlation})
                    send_notification(correlation)
                    logger.info(f"Real-time correlation found: {correlation}")

    def get_system_status(self) -> Dict[str, Any]:
        """Gets the current status of the system."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        return {
            "status": "running",
            "active_connections": len(self.active_connections),
            "events_processed": self.kafka_messages_count,
            "correlations_found": self.correlations_count,
            "event_stats": dict(self.event_stats),
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
        }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Gets the most recent events."""
        combined_events = sorted(
            list(self.recent_errors) + list(self.recent_warnings) + list(self.recent_infos),
            key=lambda e: e.timestamp,
            reverse=True
        )
        return [event.to_dict() for event in combined_events[:limit]]

    def get_recent_correlations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Gets the most recent correlations."""
        return list(self.recent_correlations)[-limit:]

    def get_metrics_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Gets the history of system metrics."""
        return [asdict(metric) for metric in list(self.metrics_history)[-limit:]]


# Global instance of the monitor
_realtime_monitor: Optional[RealtimeMonitor] = None


def get_realtime_monitor() -> Optional[RealtimeMonitor]:
    """Gets the global instance of the monitor."""
    return _realtime_monitor


def initialize_realtime_monitor(model: SentenceTransformer):
    """Initializes the global monitor."""
    global _realtime_monitor
    if _realtime_monitor is None:
        _realtime_monitor = RealtimeMonitor(model)
        logger.info("Realtime monitor initialized.")
    return _realtime_monitor
