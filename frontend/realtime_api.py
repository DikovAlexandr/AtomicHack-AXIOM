import requests
import json
import asyncio
import websockets
from datetime import datetime
from typing import Dict, List, Any
import streamlit as st
import os
import sys

# Get the base URL for the API from environment variables
# This allows for flexible configuration in different environments (e.g., Docker)
@st.cache_resource
def get_api_client():
    """Returns a cached instance of the RealtimeAPI client."""
    base_url = os.getenv("API_BASE", "http://localhost:8000")
    return RealtimeAPI(base_url)


class RealtimeAPI:
    """A class for interacting with the real-time API."""

    def __init__(self, api_base: str = "http://localhost:8000"):
        """
        Initializes the RealtimeAPI client.

        Args:
            api_base: The base URL of the backend API.
        """
        self.api_base = api_base
        self.ws_url = api_base.replace("http", "ws") + "/ws/realtime"

    def get_system_status(self) -> Dict[str, Any]:
        """Fetches the overall system status."""
        try:
            response = requests.get(f"{self.api_base}/realtime/status", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching system status: {e}")
            return {}

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches a list of recent log events."""
        try:
            response = requests.get(f"{self.api_base}/realtime/events?limit={limit}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching recent events: {e}")
            return []

    def get_recent_correlations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches a list of recent correlations."""
        try:
            response = requests.get(f"{self.api_base}/realtime/correlations?limit={limit}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching recent correlations: {e}")
            return []

    def get_metrics(self, limit: int = 50) -> Dict[str, Any]:
        """Fetches system metrics, including historical and latest."""
        try:
            response = requests.get(f"{self.api_base}/realtime/metrics?limit={limit}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching metrics: {e}")
            return {}

    def get_system_health(self) -> Dict[str, Any]:
        """Fetches a detailed health status of system components."""
        try:
            response = requests.get(f"{self.api_base}/monitoring/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching system health: {e}")
            return {}

    def get_system_info(self) -> Dict[str, Any]:
        """Fetches information about the system and application configuration."""
        try:
            response = requests.get(f"{self.api_base}/monitoring/system-info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching system info: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Fetches performance metrics."""
        try:
            response = requests.get(f"{self.api_base}/monitoring/performance", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching performance metrics: {e}")
            return {}

    def get_alerts(self) -> Dict[str, Any]:
        """Fetches active system alerts."""
        try:
            response = requests.get(f"{self.api_base}/monitoring/alerts", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")
            return {}

    def send_test_event(self) -> Dict[str, Any]:
        """Sends a test event to the backend for real-time processing."""
        try:
            response = requests.post(f"{self.api_base}/realtime/test-event", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error sending test event: {e}")
            return {}

    def process_log_realtime(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a single log entry for real-time processing."""
        try:
            response = requests.post(
                f"{self.api_base}/realtime/process-log",
                json=log_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error processing log in real-time: {e}")
            return {}


class WebSocketClient:
    """A client for handling WebSocket connections."""

    def __init__(self, ws_url: str):
        """
        Initializes the WebSocketClient.

        Args:
            ws_url: The URL of the WebSocket endpoint.
        """
        self.ws_url = ws_url
        self.websocket = None
        self.connected = False

    async def connect(self):
        """
        Connects to the WebSocket server.
        """
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            return True
        except Exception as e:
            st.error(f"Error connecting to WebSocket: {e}")
            return False

    async def disconnect(self):
        """Disconnects from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False

    async def send_message(self, message: str):
        """Sends a message to the WebSocket server."""
        if self.websocket and self.connected:
            await self.websocket.send(message)

    async def receive_message(self, timeout: float = 5.0):
        """
        Receives a message from the WebSocket server.

        Args:
            timeout: The time to wait for a message before timing out.

        Returns:
            The parsed JSON message, or None if timed out or an error occurred.
        """
        if self.websocket and self.connected:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=timeout
                )
                return json.loads(message)
            except asyncio.TimeoutError:
                return None
            except Exception as e:
                st.error(f"Error receiving WebSocket message: {e}")
                return None
        return None


def format_timestamp(timestamp_str: str) -> str:
    """Formats an ISO timestamp string into H:M:S format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return timestamp_str


def format_event_type(event_type: str) -> str:
    """Formats an event type with a corresponding icon."""
    type_icons = {
        "ERROR": "🔴",
        "WARNING": "🟡",
        "INFO": "🔵",
        "CORRELATION": "🔗"
    }
    return f"{type_icons.get(event_type, '⚪')} {event_type}"


def get_status_color(status: str) -> str:
    """Returns a status color icon based on the status string."""
    status_colors = {
        "healthy": "🟢",
        "warning": "🟡",
        "critical": "🔴",
        "unknown": "⚪"
    }
    return status_colors.get(status.lower(), "⚪")


def format_bytes(bytes_value: float) -> str:
    """Formats a byte value into a human-readable string (KB, MB, GB, etc.)."""
    if not isinstance(bytes_value, (int, float)) or bytes_value < 0:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"
