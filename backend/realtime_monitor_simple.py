"""
Simple real-time monitor for standalone mode.
This module provides a mock implementation of the real-time monitor for UI testing
and standalone operation without a full backend environment.
"""

import asyncio
import json
import logging
import random
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional

import psutil
from fastapi import WebSocket

logger = logging.getLogger(__name__)

_simple_realtime_monitor: Optional["SimpleRealtimeMonitor"] = None


class SimpleRealtimeEvent:
    """Data class for a mock real-time event."""

    def __init__(self, timestamp: datetime, event_type: str, message: str, component: str,
                 file_name: str = "mock.log", line_number: int = 0):
        self.timestamp = timestamp
        self.event_type = event_type
        self.message = message
        self.component = component
        self.file_name = file_name
        self.line_number = line_number

    def to_dict(self):
        """Serializes the event to a dictionary."""
        return self.__dict__


class SimpleSystemMetrics:
    """
    Data class for mock system metrics.
    """

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.cpu_percent = psutil.cpu_percent(interval=None)
        self.memory_percent = psutil.virtual_memory().percent
        self.disk_usage_percent = psutil.disk_usage('/').percent
        self.network_io = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        self.process_count = len(psutil.pids())

    def to_dict(self):
        """Serializes the metrics to a dictionary."""
        return self.__dict__


class SimpleRealtimeMonitor:
    """
    A mock real-time monitor that generates random events and metrics
    for testing and standalone demonstrations.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.recent_events: deque = deque(maxlen=100)
        self.recent_correlations: deque = deque(maxlen=20)
        self.metrics_history: deque = deque(maxlen=50)
        self.event_stats: defaultdict = defaultdict(int)
        self.kafka_messages_count = 0
        self.correlations_count = 0
        self._metrics_task: Optional[asyncio.Task] = None
        self._start_metrics_collection()
        logger.info("SimpleRealtimeMonitor initialized.")

    async def connect_websocket(self, websocket: WebSocket):
        """Accepts and stores a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client.host}")

    async def disconnect_websocket(self, websocket: WebSocket):
        """Removes a disconnected WebSocket."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client.host}")

    async def broadcast_json(self, data: dict):
        """Broadcasts a JSON message to all connected clients."""
        message = json.dumps(data, default=str)
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except Exception:
                self.active_connections.remove(connection)

    def process_realtime_log(self, log_data: Dict[str, Any]) -> SimpleRealtimeEvent:
        """Generates a mock event from log data."""
        event = SimpleRealtimeEvent(
            timestamp=datetime.now(),
            event_type=random.choice(["ERROR", "WARNING", "INFO"]),
            message=log_data.get('raw', 'Mock log message'),
            component=log_data.get('source', 'mock_component'),
            file_name=log_data.get('file_name', 'mock.log'),
            line_number=log_data.get('line_number', 0),
        )
        self.recent_events.append(event)
        self.event_stats[event.event_type] += 1
        self.kafka_messages_count += 1
        return event

    async def find_correlations_realtime(self, new_event: SimpleRealtimeEvent) -> List[Dict[str, Any]]:
        """Generates a mock correlation with a 50% probability for ERROR events."""
        if new_event.event_type == "ERROR" and random.random() < 0.5:
            correlation = {
                "anomaly_id": random.randint(1, 10),
                "anomaly_text": "Mock Anomaly",
                "problem_id": random.randint(1, 10),
                "problem_text": "Mock Problem",
                "log": new_event.message,
                "similarity": random.uniform(0.7, 0.95),
                "timestamp": datetime.now().isoformat()
            }
            self.recent_correlations.append(correlation)
            self.correlations_count += 1
            return [correlation]
        return []

    async def _collect_metrics_periodically(self):
        """Periodically collects and broadcasts mock system metrics."""
        while True:
            await asyncio.sleep(5)
            metrics = SimpleSystemMetrics()
            self.metrics_history.append(metrics)
            await self.broadcast_json({"type": "metrics", "data": metrics.to_dict()})

    def _start_metrics_collection(self):
        """Starts the background task for metrics collection if not already running."""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self._collect_metrics_periodically())

    def get_system_status(self) -> Dict[str, Any]:
        """Gets the current mock status of the system."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else SimpleSystemMetrics()
        return {
            **latest_metrics.to_dict(),
            "active_websocket_connections": len(self.active_connections),
            "kafka_messages_processed": self.kafka_messages_count,
            "correlations_found": self.correlations_count
        }

    def get_recent_events(self, limit: int) -> List[Dict[str, Any]]:
        """Gets the most recent mock events."""
        return [event.to_dict() for event in list(self.recent_events)[-limit:]]

    def get_recent_correlations(self, limit: int) -> List[Dict[str, Any]]:
        """Gets the most recent mock correlations."""
        return list(self.recent_correlations)[-limit:]

    def get_metrics_history(self, limit: int) -> List[Dict[str, Any]]:
        """Gets the history of mock system metrics."""
        return [m.to_dict() for m in list(self.metrics_history)[-limit:]]


def initialize_simple_realtime_monitor():
    """Initializes the global simple monitor instance."""
    global _simple_realtime_monitor
    if _simple_realtime_monitor is None:
        _simple_realtime_monitor = SimpleRealtimeMonitor()


def get_simple_realtime_monitor() -> Optional[SimpleRealtimeMonitor]:
    """Gets the global instance of the simple monitor."""
    return _simple_realtime_monitor
