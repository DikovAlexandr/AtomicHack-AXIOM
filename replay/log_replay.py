import os
import asyncio
import json
import glob
import re
import logging
from datetime import datetime, timezone
from dateutil import parser as dtparser

from aiokafka import AIOKafkaProducer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_PREFIX = os.getenv("TOPIC_PREFIX", "logs")
LOG_DIR = os.getenv("LOG_DIR", "/logs")
SPEED = float(os.getenv("SPEED", "1.0"))
JITTER_MS = int(os.getenv("JITTER_MS", "0"))

logger.info(f"KAFKA_BOOTSTRAP: {KAFKA_BOOTSTRAP}")
logger.info(f"TOPIC_PREFIX: {TOPIC_PREFIX}")
logger.info(f"LOG_DIR: {LOG_DIR}")
logger.info(f"SPEED: {SPEED}")

line_re = re.compile(
    r"^(?P<ts>[\d\-T:\.]+) (?P<level>INFO|ERROR|WARNING) (?P<component>[^:]+): (?P<message>.+)$"
)

def iter_log_lines():
    events = []
    search_path = os.path.join(LOG_DIR, "*.txt")
    log_files = glob.glob(search_path)
    
    logger.info(f"Searching for logs in '{LOG_DIR}'. Found {len(log_files)} files.")
    if not log_files:
        if os.path.exists(LOG_DIR):
            logger.warning(f"Directory '{LOG_DIR}' exists, but contains no .txt/.log files. Content: {os.listdir(LOG_DIR)}")
        else:
            logger.error(f"Directory '{LOG_DIR}' does not exist inside the container!")

    for path in sorted(log_files + glob.glob(os.path.join(LOG_DIR, "*.log"))):
        fname = os.path.basename(path)
        logger.info(f"Reading file: {fname}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines_read = 0
                for idx, raw in enumerate(f, start=1):
                    lines_read += 1
                    raw = raw.strip()
                    if not raw:
                        continue
                    m = line_re.match(raw)
                    if not m:
                        logger.warning(f"Line {idx} in {fname} did not match pattern: '{raw}'")
                        continue
                    ts = dtparser.parse(m.group("ts"))
                    ev = {
                        "timestamp": ts.isoformat(),
                        "level": m.group("level"),
                        "component": m.group("component"),
                        "message": m.group("message"),
                        "file": fname,
                        "line": idx,
                        "raw": raw,
                        "topic": f"{TOPIC_PREFIX}.{os.path.splitext(fname)[0]}",
                    }
                    events.append(ev)
            logger.info(f"Finished reading {fname}, processed {lines_read} lines, found {len(events)} valid events so far.")
        except Exception as e:
            logger.error(f"Failed to read or process file {path}: {e}")


    if events:
        events.sort(key=lambda e: e["timestamp"])
        logger.info(f"Total valid events found across all files: {len(events)}")
    return events

async def main():
    logger.info("Log replay service started.")
    events = iter_log_lines()
    if not events:
        logger.warning("No log events found to replay. Service will exit.")
        return

    while True:
        t0_log = datetime.fromisoformat(events[0]["timestamp"])
        t0_wall = datetime.now(tz=timezone.utc)

        producer = None
        try:
            logger.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP}...")
            producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                request_timeout_ms=30000,
                metadata_max_age_ms=60000,
            )
            await producer.start()
            logger.info("Kafka producer started successfully.")

            for i, ev in enumerate(events, 1):
                event_copy = ev.copy()
                
                ts_log = datetime.fromisoformat(ev["timestamp"])
                delta_log = (ts_log - t0_log).total_seconds()
                due_wall = t0_wall.timestamp() + delta_log / max(SPEED, 0.0001)
                sleep_s = due_wall - datetime.now(tz=timezone.utc).timestamp()
                
                if JITTER_MS:
                    import random
                    sleep_s += random.uniform(0, JITTER_MS/1000.0)
                
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                topic = ev.pop("topic")
                payload = json.dumps(ev, ensure_ascii=False).encode("utf-8")
                
                logger.debug(f"Sending event {i}/{len(events)} to topic '{topic}'")
                await producer.send_and_wait(topic, payload)
                logger.info(f"→ {topic}: {event_copy['timestamp']} {event_copy['level']} {event_copy['component']} {event_copy['message']}")
        
        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
            logger.info("Reconnecting in 10 seconds...")
            await asyncio.sleep(10)

        finally:
            if producer:
                logger.info("Stopping Kafka producer...")
                await producer.stop()
                logger.info("Kafka producer stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)