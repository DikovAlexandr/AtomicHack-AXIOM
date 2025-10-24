"""
Core processing logic for log analysis, including embedding generation,
caching, and correlation finding.
"""
import io
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import redis
import torch
from sentence_transformers import SentenceTransformer, util

# --- Constants ---
CORRELATION_WINDOW = timedelta(hours=24)
SIMILARITY_THRESHOLD = 0.9
REFERENCE_SIMILARITY_THRESHOLD = 0.999
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'microsoft/codebert-base'
TENSOR_DTYPE = np.float32
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Redis Connection ---
_redis_connection = None


def get_redis_connection() -> Optional[redis.Redis]:
    """
    Establishes and retrieves a Redis connection, with a simple ping-based health check.

    Returns:
        An active Redis connection object or None if the connection fails.
    """
    global _redis_connection
    if _redis_connection:
        try:
            _redis_connection.ping()
            return _redis_connection
        except redis.exceptions.ConnectionError:
            logger.warning("Redis connection lost, re-establishing.")
            _redis_connection = None

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
        r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}.")
        _redis_connection = r
        return r
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}. Caching will be disabled.")
        _redis_connection = None
        return None

def serialize_embedding(embedding: torch.Tensor) -> bytes:
    """Serializes a torch.Tensor to bytes for Redis storage."""
    return embedding.cpu().numpy().astype(TENSOR_DTYPE).tobytes()


def deserialize_embedding(embedding_bytes: bytes) -> torch.Tensor:
    """Deserializes bytes from Redis back into a torch.Tensor."""
    numpy_array = np.frombuffer(embedding_bytes, dtype=TENSOR_DTYPE).copy()
    return torch.from_numpy(numpy_array).to(DEVICE)


def get_or_compute_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    r_conn: Optional[redis.Redis]
) -> torch.Tensor:
    """
    Retrieves embeddings from cache or computes them if not found.

    Args:
        texts: A list of strings to get embeddings for.
        model: The SentenceTransformer model to use for computation.
        r_conn: An optional Redis connection for caching.

    Returns:
        A torch.Tensor containing the embeddings for all texts.
    """
    if not texts:
        return torch.tensor([], device=DEVICE)

    final_embeddings: List[Optional[torch.Tensor]] = [None] * len(texts)
    missed_indices, texts_to_compute = [], []

    if r_conn:
        keys = [f"embedding:{text}" for text in texts]
        cached_results = r_conn.mget(keys)
        for i, result in enumerate(cached_results):
            if result:
                final_embeddings[i] = deserialize_embedding(result)
            else:
                missed_indices.append(i)
                texts_to_compute.append(texts[i])
        logger.info(f"Cache hit for {len(texts) - len(texts_to_compute)}/{len(texts)} embeddings.")
    else:
        texts_to_compute = texts
        missed_indices = list(range(len(texts)))

    if texts_to_compute:
        logger.info(f"Computing {len(texts_to_compute)} new embeddings.")
        with torch.inference_mode():
            new_embeddings = model.encode(texts_to_compute, convert_to_tensor=True, device=DEVICE)

        for i, new_emb in zip(missed_indices, new_embeddings):
            final_embeddings[i] = new_emb
        
        if r_conn:
            pipe = r_conn.pipeline()
            for i in missed_indices:
                pipe.set(f"embedding:{texts[i]}", serialize_embedding(final_embeddings[i]))
            pipe.execute()
    
    return torch.stack(final_embeddings)


# --- Text and Log Parsing ---

def preprocess_message(message: str) -> str:
    """
    Cleans and standardizes a log message for embedding.

    Args:
        message: The raw log message string.

    Returns:
        A cleaned and standardized string.
    """
    text = message.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single log line into a structured dictionary.

    Args:
        line: The log line string.

    Returns:
        A dictionary with parsed components or None if parsing fails.
    """
    try:
        parts = line.strip().split(' ', 3)
        if len(parts) < 4 or parts[1] not in ["ERROR", "WARNING"]:
            return None
        return {
            "timestamp": datetime.fromisoformat(parts[0]),
            "type": parts[1],
            "component": parts[2].rstrip(':'),
            "message": parts[3],
            "full_line": line.strip()
        }
    except (ValueError, IndexError):
        return None


# --- Core Correlation Logic ---

def _prepare_reference_data(csv_content: str, model: SentenceTransformer, r_conn: Optional[redis.Redis]) -> dict:
    """Loads, processes, and embeds the reference anomaly/problem data."""
    df = pd.read_csv(io.StringIO(csv_content), sep=';')
    df.rename(columns={
        'ID проблемы': 'problem_id', 'Проблема': 'problem_text',
        'ID аномалии': 'anomaly_id', 'Аномалия': 'anomaly_text'
    }, inplace=True)

    problems_df = df.groupby('problem_text', as_index=False)['problem_id'].min().rename(columns={'problem_id': 'id', 'problem_text': 'text'})
    anomalies_df = df.groupby('anomaly_text', as_index=False)['anomaly_id'].min().rename(columns={'anomaly_id': 'id', 'anomaly_text': 'text'})
    
    problem_embeddings = get_or_compute_embeddings([preprocess_message(t) for t in problems_df['text']], model, r_conn)
    anomaly_embeddings = get_or_compute_embeddings([preprocess_message(t) for t in anomalies_df['text']], model, r_conn)
    logger.info(f"Reference embeddings ready: {len(problems_df)} problems, {len(anomalies_df)} anomalies.")

    return {
        "problems_df": problems_df,
        "anomalies_df": anomalies_df,
        "problem_embeddings": problem_embeddings,
        "anomaly_embeddings": anomaly_embeddings,
        "problem_id_to_text": problems_df.set_index('id')['text'].to_dict(),
        "anomaly_id_to_text": anomalies_df.set_index('id')['text'].to_dict(),
        "valid_pairs_df": df
    }

def _parse_and_embed_logs(log_files_data: List[Tuple[str, str]], model: SentenceTransformer, r_conn: Optional[redis.Redis]) -> List[Dict[str, Any]]:
    """Parses all log files and computes embeddings for each log message."""
    all_events = []
    for file_name, content in log_files_data:
        for line_num, line in enumerate(content.splitlines(), 1):
            if parsed_event := parse_log_line(line):
                parsed_event.update({"file": file_name, "line_num": line_num})
                all_events.append(parsed_event)
    
    if not all_events:
        return []

    cleaned_messages = [preprocess_message(event['message']) for event in all_events]
    all_embeddings = get_or_compute_embeddings(cleaned_messages, model, r_conn)
    for i, event in enumerate(all_events):
        event['embedding'] = all_embeddings[i]
        
    return all_events

def _find_correlations(error_events: List[Dict], warning_events: List[Dict], ref_data: dict) -> List[Dict]:
    """Finds and validates correlations between error and warning events."""
    if not error_events or not warning_events:
        return []

    error_matrix = torch.stack([e['embedding'] for e in error_events])
    warning_matrix = torch.stack([w['embedding'] for w in warning_events])
    
    # Find best problem/anomaly matches for all errors/warnings
    err_problem_sims = util.cos_sim(error_matrix, ref_data["problem_embeddings"])
    _, best_problem_indices = torch.max(err_problem_sims, dim=1)
    
    warn_anomaly_sims = util.cos_sim(warning_matrix, ref_data["anomaly_embeddings"])
    best_anomaly_sims, best_anomaly_indices = torch.max(warn_anomaly_sims, dim=1)
    
    problem_ids = torch.tensor(ref_data["problems_df"]['id'].values, device=DEVICE)[best_problem_indices]
    anomaly_ids = torch.tensor(ref_data["anomalies_df"]['id'].values, device=DEVICE)[best_anomaly_indices]

    # Filter pairs by time and similarity
    similarity_matrix = util.cos_sim(error_matrix, warning_matrix)
    err_ts = torch.tensor([e['timestamp'].timestamp() for e in error_events], device=DEVICE)
    warn_ts = torch.tensor([w['timestamp'].timestamp() for w in warning_events], device=DEVICE)
    time_diff_matrix = torch.abs(err_ts.unsqueeze(1) - warn_ts.unsqueeze(0))
    
    mask = (similarity_matrix >= SIMILARITY_THRESHOLD) & (time_diff_matrix <= CORRELATION_WINDOW.total_seconds())
    candidate_indices = torch.nonzero(mask)

    if candidate_indices.shape[0] == 0:
        return []

    # Validate pairs against the reference CSV
    valid_pairs_df = ref_data["valid_pairs_df"]
    correlated_issues = set()
    for err_idx, warn_idx in candidate_indices.tolist():
        problem_id = problem_ids[err_idx].item()
        anomaly_id = anomaly_ids[warn_idx].item()

        is_valid = not valid_pairs_df[
            (valid_pairs_df['problem_id'] == problem_id) & 
            (valid_pairs_df['anomaly_id'] == anomaly_id)
        ].empty

        if is_valid:
            error_event = error_events[err_idx]
            corr_tuple = (
                anomaly_id, ref_data["anomaly_id_to_text"].get(anomaly_id),
                problem_id, ref_data["problem_id_to_text"].get(problem_id),
                error_event["file"], error_event["line_num"], error_event["full_line"]
            )
            correlated_issues.add(corr_tuple)
            
    return [dict(zip(["anomaly_id", "anomaly_text", "problem_id", "problem_text", "file_name", "line_number", "log"], item)) for item in correlated_issues]


def process_log_data(
    anomalies_problems_csv_content: str,
    log_files_data: List[Tuple[str, str]],
    model: SentenceTransformer
) -> List[Dict[str, Any]]:
    """
    Main function to process log data, find correlations, and return results.

    Args:
        anomalies_problems_csv_content: String content of the reference CSV.
        log_files_data: A list of tuples, each with a filename and its content.
        model: The SentenceTransformer model.

    Returns:
        A list of dictionaries, each representing a found correlation.
    """
    start_time = time.time()
    r_conn = get_redis_connection()

    try:
        ref_data = _prepare_reference_data(anomalies_problems_csv_content, model, r_conn)
    except Exception as e:
        logger.error(f"Could not read or process anomalies/problems CSV: {e}")
        return []

    all_events = _parse_and_embed_logs(log_files_data, model, r_conn)
    if not all_events:
        logger.warning("No relevant ERROR/WARNING events found in logs.")
        return []

    error_events = [e for e in all_events if e['type'] == 'ERROR']
    warning_events = [w for w in all_events if w['type'] == 'WARNING']
    logger.info(f"Found {len(all_events)} events: {len(error_events)} ERRORs, {len(warning_events)} WARNINGs.")

    correlated_issues = _find_correlations(error_events, warning_events, ref_data)
    
    logger.info(f"Found {len(correlated_issues)} unique correlations in {time.time() - start_time:.2f} seconds.")
    
    return sorted(correlated_issues, key=lambda x: (x['problem_text'], x['file_name']))


def find_best_match_id_by_embedding(message_embedding: torch.Tensor, mapping_df: pd.DataFrame) -> Tuple[int, float]:
    """
    Finds the best matching ID from a DataFrame based on embedding similarity.

    Args:
        message_embedding: The embedding of the message to match.
        mapping_df: DataFrame with 'id' and 'embedding' columns.

    Returns:
        A tuple of the best match ID and the similarity score, or (-1, 0.0) if no match.
    """
    if mapping_df.empty or 'embedding' not in mapping_df.columns:
        return -1, 0.0
        
    candidate_embeddings = torch.stack(mapping_df['embedding'].tolist()).to(DEVICE)
    similarities = util.cos_sim(message_embedding, candidate_embeddings)[0]
    best_match_index = torch.argmax(similarities).item()
    max_similarity = similarities[best_match_index].item()
    
    if max_similarity >= REFERENCE_SIMILARITY_THRESHOLD:
        best_match_id = mapping_df.iloc[best_match_index]['id']
        return best_match_id, max_similarity
        
    return -1, max_similarity