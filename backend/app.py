import io
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import psutil
from fastapi import (FastAPI, UploadFile, File, Request, HTTPException, WebSocket,
                   WebSocketDisconnect)
from fastapi.responses import StreamingResponse, JSONResponse
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from processing import process_log_data, MODEL_NAME, DEVICE

# Optional imports
try:
    from telegram_bot import send_notification
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    def send_notification(data):
        logger.info(f"Telegram notification (disabled): {data}")

try:
    from kafka import KafkaAdminClient
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Removed Triton-related imports and logic
REALTIME_AVAILABLE = False  # Default to False

def check_kafka_connection(bootstrap_servers):
    if not KAFKA_AVAILABLE:
        logger.warning("Kafka client library not installed. Real-time features disabled.")
        return False
    try:
        client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='backend-health-check',
            request_timeout_ms=5000  # 5 seconds timeout
        )
        # list_topics is a lightweight way to check the connection
        client.list_topics()
        logger.info(f"Successfully connected to Kafka at {bootstrap_servers}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Kafka at {bootstrap_servers}: {e}")
        return False
    finally:
        if 'client' in locals() and client:
            client.close()

try:
    from realtime_monitor import initialize_realtime_monitor, get_realtime_monitor
    REALTIME_AVAILABLE = True
    logger.info("✅ Realtime monitor module available")
except ImportError as e:
    REALTIME_AVAILABLE = False
    logger.warning(f"⚠️ Realtime monitor not available: {e}")
    def initialize_realtime_monitor(model):
        logger.info("Realtime monitor not available")
    def get_realtime_monitor():
        return None

# try:
#     from realtime_monitor_simple import initialize_simple_realtime_monitor, get_simple_realtime_monitor
#     SIMPLE_REALTIME_AVAILABLE = True
#     logger.info("✅ Simple realtime monitor module available")
# except ImportError as e:
#     SIMPLE_REALTIME_AVAILABLE = False
#     logger.warning(f"⚠️ Simple realtime monitor not available: {e}")
#     def initialize_simple_realtime_monitor():
#         logger.info("Simple realtime monitor not available")
#     def get_simple_realtime_monitor():
#         return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, handling startup and shutdown events.
    """
    global REALTIME_AVAILABLE
    # Set default environment variables
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    os.environ.setdefault("KAFKA_BOOTSTRAP", "localhost:9092")
    os.environ.setdefault("ANOMALIES_CSV_PATH", "data/anomalies_problems.csv")
    os.environ.setdefault("DB_PATH", "data/db/subscribers.db")
    
    logger.info("🚀 Starting STANDALONE backend initialization...")
    
    # Check for Kafka connection
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP")
    REALTIME_AVAILABLE = check_kafka_connection(kafka_bootstrap_servers)

    # Load the model with error handling
    app.state.model = None
    try:
        logger.info(f"Loading model '{MODEL_NAME}' on device '{DEVICE}'...")
        app.state.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.warning("🔄 Running without ML model (standalone mode)...")
        app.state.model = None

    # Initialize real-time monitor (optional)
    if REALTIME_AVAILABLE and app.state.model:
        try:
            initialize_realtime_monitor(app.state.model)
            logger.info("✅ Realtime monitor initialized.")
        except Exception as e:
            logger.error(f"❌ Error initializing realtime monitor: {e}")
    else:
        logger.info("ℹ️ Realtime monitor not available or no model")
    
    # # Initialize simple monitor for standalone mode
    # if SIMPLE_REALTIME_AVAILABLE:
    #     try:
    #         initialize_simple_realtime_monitor()
    #         logger.info("✅ Simple realtime monitor initialized.")
    #     except Exception as e:
    #         logger.error(f"❌ Error initializing simple realtime monitor: {e}")
    # else:
    #     logger.info("ℹ️ Simple realtime monitor not available")
    
    logger.info("🔍 Checking external services...")
    logger.info(f"  - Telegram: {'Available' if TELEGRAM_AVAILABLE else '❌ Not available'}")
    logger.info(f"  - Realtime (Kafka): {'✅ Available' if REALTIME_AVAILABLE else '❌ Not available'}")
    logger.info(f"  - ML Model: {'✅ Available' if app.state.model else '❌ Not available'}")
    
    logger.info(f"Backend API running at http://{os.getenv('UVICORN_HOST', '0.0.0.0')}:{os.getenv('UVICORN_PORT', '8000')}")
    logger.info("Standalone backend startup completed!")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Log Correlation Service",
    description="An API to analyze log files and find correlations between anomalies and problems.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=200, tags=["Health Check"])
async def health_check():
    """
    Provides a health check endpoint for monitoring the service status.
    """
    return JSONResponse(content={
        "status": "ok",
        "mode": "standalone",
        "services": {
            "telegram": TELEGRAM_AVAILABLE,
            "realtime": REALTIME_AVAILABLE,
            "ml_model": app.state.model is not None
        },
        "timestamp": datetime.now().isoformat()
    })


@app.post("/analyze-logs/",
          summary="Analyze logs to find correlations",
          description="Upload an 'anomalies_problems.csv' file and one or more log files (.txt) "
                      "to find correlations between ERROR and WARNING events.")
async def analyze_logs_endpoint(
    request: Request,
    anomalies_file: UploadFile = File(..., description="The CSV file with anomaly-problem mappings."),
    log_files: List[UploadFile] = File(..., description="One or more text log files to analyze.")
):
    """
    Analyzes uploaded log files against an anomaly definition file to find correlations.
    """
    logger.info("=" * 50)
    logger.info("🚀 ANALYZE LOGS ENDPOINT CALLED")
    logger.info(f"Anomalies file: {anomalies_file.filename}")
    for i, log_file in enumerate(log_files):
        logger.info(f"Log file {i}: {log_file.filename}")

    if not anomalies_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Anomalies file must be a .csv file.")
    if not all(f.filename.endswith('.txt') for f in log_files):
        raise HTTPException(status_code=400, detail="All log files must be .txt files.")

    try:
        anomalies_csv_content = (await anomalies_file.read()).decode('utf-8')
        log_files_data = [(f.filename, (await f.read()).decode('utf-8')) for f in log_files]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded files: {e}")

    logger.info("🔧 Starting log processing...")
    model = request.app.state.model
    
    if not model:
        logger.error("❌ ML model not available. Cannot perform analysis.")
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded, so analysis cannot be performed. Please check the backend startup logs."
        )

    correlation_results = process_log_data(
        anomalies_problems_csv_content=anomalies_csv_content,
        log_files_data=log_files_data,
        model=model,
    )

    logger.info(f"📊 Processing completed. Found {len(correlation_results)} correlations")

    for result in correlation_results:
        send_notification(result)

    results_df = pd.DataFrame(correlation_results)
    stream = io.StringIO()
    columns_order = ["anomaly_id", "anomaly_text", "problem_id", "problem_text", "file_name", "line_number", "log"]
    results_df.to_csv(stream, index=False, columns=[c for c in columns_order if c in results_df.columns], encoding='utf-8-sig')

    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=correlation_results.csv"
    return response


# ==================== REAL-TIME MONITORING ENDPOINTS ====================

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring updates.
    """
    monitor = get_realtime_monitor()
    if not monitor:
        await websocket.close(code=1000, reason="Realtime monitor not available or initialized")
        return
    
    await monitor.connect_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        await monitor.disconnect_websocket(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await monitor.disconnect_websocket(websocket)


@app.post("/realtime/process-log", tags=["Real-time Monitoring"])
async def process_realtime_log(log_data: Dict[str, Any]):
    """
    Processes a single log entry in real-time.
    """
    monitor = get_realtime_monitor()
    if not monitor:
        raise HTTPException(status_code=503, detail="Realtime monitor not initialized")
    
    try:
        event = monitor.process_realtime_log(log_data)
        if not event:
            return {"status": "ignored", "reason": "Not a valid ERROR/WARNING event"}
        
        await monitor.broadcast_event(event)
        
        return {"status": "processed", "event": event.__dict__}
    except Exception as e:
        logger.error(f"Error processing real-time log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/status", tags=["Real-time Monitoring"])
async def get_realtime_status():
    """
    Gets the current status of the real-time monitoring system.
    """
    monitor = get_realtime_monitor()
    if monitor:
        return monitor.get_system_status()
    
    return {
        "status": "standalone",
        "message": "Real-time monitor not active. Using fallback metrics.",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }


@app.get("/realtime/events", tags=["Real-time Monitoring"])
async def get_recent_events(limit: int = 50):
    """
    Retrieves the most recent log events.
    """
    monitor = get_realtime_monitor()
    if monitor:
        return monitor.get_recent_events(limit)
    
    return []


@app.get("/realtime/correlations", tags=["Real-time Monitoring"])
async def get_recent_correlations(limit: int = 20):
    """
    Retrieves the most recent correlations found.
    """
    monitor = get_realtime_monitor()
    if monitor:
        return monitor.get_recent_correlations(limit)

    return []


@app.get("/realtime/metrics", tags=["Real-time Monitoring"])
async def get_metrics(limit: int = 50):
    """
    Retrieves historical and the latest system metrics.
    """
    monitor = get_realtime_monitor()
    if monitor:
        return {
            "metrics": monitor.get_metrics_history(limit),
            "latest": monitor.metrics_history[-1] if monitor.metrics_history else None
        }
    
    return {"metrics": [], "latest": None}


@app.post("/realtime/test-event", tags=["Real-time Monitoring"])
async def test_realtime_event():
    """
    Sends a test event to the real-time monitoring system.
    """
    monitor = get_realtime_monitor()
    if not monitor:
        raise HTTPException(status_code=503, detail="No active real-time monitor to send a test event.")
    
    test_event_data = {
        "timestamp": datetime.now(),
        "event_type": "ERROR",
        "message": "This is a test error message for real-time monitoring.",
        "component": "test-component",
        "file_name": "test.log",
        "line_number": 123
    }

    if hasattr(monitor, 'process_realtime_log'):
        event = monitor.process_realtime_log({"raw": "test log line"})  # Simplified
        if event:
            await monitor.broadcast_event(event)
            return {"status": "test_event_sent", "event": event.__dict__}
    
    raise HTTPException(status_code=500, detail="Failed to create or broadcast a test event.")

# ==================== SYSTEM MONITORING ENDPOINTS ====================

@app.get("/monitoring/health", tags=["System Monitoring"])
async def detailed_health_check():
    """
    Provides a detailed health check of the system and its components.
    """
    monitor = get_realtime_monitor()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "realtime_monitor": "healthy" if monitor else "unavailable",
            "model": "healthy" if app.state.model else "unavailable",
        }
    }
    
    if not monitor or not app.state.model:
        health_status["status"] = "degraded"
        
    return health_status


@app.get("/monitoring/system-info", tags=["System Monitoring"])
async def get_system_info():
    """
    Retrieves detailed information about the system hardware and application configuration.
    """
    return {
        "system": {
            "platform": psutil.os.uname().version,
            "python_version": psutil.sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
        },
        "application": {
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "redis_host": os.getenv("REDIS_HOST"),
            "kafka_bootstrap": os.getenv("KAFKA_BOOTSTRAP")
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/monitoring/performance", tags=["System Monitoring"])
async def get_performance_metrics():
    """
    Retrieves a summary of the system's performance metrics.
    """
    monitor = get_realtime_monitor()
    
    if monitor and hasattr(monitor, 'metrics_history') and monitor.metrics_history:
        latest_metrics = monitor.metrics_history[-1]
        return {
            "current_metrics": latest_metrics.__dict__,
            "performance_summary": {
                "events_per_minute": getattr(monitor, 'kafka_messages_count', 0),
                "correlations_per_minute": getattr(monitor, 'correlations_count', 0),
                "active_connections": len(getattr(monitor, 'active_connections', [])),
            },
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "current_metrics": {},
        "performance_summary": {},
        "timestamp": datetime.now().isoformat(),
        "mode": "standalone"
    }


@app.get("/monitoring/alerts", tags=["System Monitoring"])
async def get_system_alerts():
    """
    Generates system alerts based on the latest metrics.
    """
    monitor = get_realtime_monitor()
    alerts = []
    
    if monitor and hasattr(monitor, 'metrics_history') and monitor.metrics_history:
        latest_metrics = monitor.metrics_history[-1]
        
        if latest_metrics.cpu_percent > 80:
            alerts.append({"type": "warning", "message": f"High CPU usage: {latest_metrics.cpu_percent:.1f}%"})
        
        if latest_metrics.memory_percent > 85:
            alerts.append({"type": "critical", "message": f"High memory usage: {latest_metrics.memory_percent:.1f}%"})
    
    return {
        "alerts": alerts,
        "alert_count": len(alerts),
        "status": "healthy" if not alerts else "critical" if any(a['type'] == 'critical' for a in alerts) else "warning"
    }