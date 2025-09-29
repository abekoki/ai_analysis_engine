"""
File operation utilities
"""

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .logger import get_logger
from .exploration_utils import extract_json_with_llm

logger = get_logger(__name__)


def read_markdown(file_path: str) -> str:
    """
    Read markdown file content

    Args:
        file_path: Path to markdown file

    Returns:
        Content of the markdown file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read markdown file {file_path}: {e}")
        raise


def read_csv_safe(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Safely read CSV file with error handling

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame with CSV data
    """
    try:
        # Default parameters for CSV reading
        default_kwargs = {
            'encoding': 'utf-8',
            'low_memory': False
        }
        default_kwargs.update(kwargs)

        df = pd.read_csv(file_path, **default_kwargs)
        logger.info(f"Successfully read CSV {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV file {file_path}: {e}")
        raise


def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """
    Write content to file

    Args:
        file_path: Path to output file
        content: Content to write
        encoding: File encoding
    """
    try:
        ensure_directory(str(Path(file_path).parent))
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        logger.info(f"Successfully wrote to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {e}")
        raise


def ensure_directory(dir_path: str) -> None:
    """
    Ensure directory exists, create if not

    Args:
        dir_path: Directory path
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text using LLM

    Args:
        text: Text containing JSON

    Returns:
        Extracted JSON object or None
    """
    try:
        # Use LLM-based extraction only
        result = extract_json_with_llm(text)
        if result is not None:
            logger.info("Successfully extracted JSON from text using LLM")
            return result

        return None
    except Exception as e:
        raise RuntimeError(f"LLM-based JSON extraction failed: {e}") from e


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    return text


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap

    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If not the last chunk, try to find a good breaking point
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            last_period = text.rfind('.', end - 100, end)
            last_newline = text.rfind('\n', end - 100, end)

            # Use the latest good breaking point
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size // 2:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap

        # Ensure progress
        if start >= end:
            start = end

    return chunks


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = path.stat()

    return {
        'path': str(path.absolute()),
        'name': path.name,
        'extension': path.suffix,
        'size': stat.st_size,
        'modified_time': stat.st_mtime,
        'is_file': path.is_file(),
        'is_dir': path.is_dir()
    }


def ensure_analysis_output_structure(base_dir: Path) -> Dict[str, Path]:
    """Ensure standard directory structure for analysis outputs exists.

    Args:
        base_dir: Base directory for the current analysis run

    Returns:
        Dictionary with resolved paths for key sub-directories
    """

    base_path = Path(base_dir)
    results_dir = base_path / "results"
    reports_dir = results_dir / "reports"
    images_dir = base_path / "images"
    logs_dir = base_path / "logs"
    langgraph_dir = logs_dir / "langgraph_context"

    for path in (results_dir, reports_dir, images_dir, logs_dir, langgraph_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_path,
        "results": results_dir,
        "reports": reports_dir,
        "images": images_dir,
        "logs": logs_dir,
        "langgraph": langgraph_dir,
    }


def get_dataset_output_dirs(base_dir: Path, dataset_id: str) -> Dict[str, Path]:
    """Return dataset-specific output directories, creating them if necessary."""

    structure = ensure_analysis_output_structure(base_dir)
    dataset_images = structure["images"] / dataset_id
    dataset_reports = structure["reports"]
    dataset_images.mkdir(parents=True, exist_ok=True)
    return {
        "images": dataset_images,
        "reports": dataset_reports,
    }


def filter_dataframe_by_interval(df: pd.DataFrame, interval: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Filter dataframe rows to the specified start/end interval if possible."""

    if df is None or df.empty or not interval:
        return df

    start = interval.get("start") or interval.get("start_frame")
    end = interval.get("end") or interval.get("end_frame")
    if start is None or end is None:
        return df

    filtered = df.copy()
    column_candidates: Iterable[str] = ("frame", "frame_num", "timestamp", "time")

    for column in column_candidates:
        if column not in filtered.columns:
            continue

        series = filtered[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            start_val = pd.to_datetime(start, errors="coerce")
            end_val = pd.to_datetime(end, errors="coerce")
        else:
            start_val = start
            end_val = end

        mask = (series >= start_val) & (series <= end_val)
        filtered = filtered.loc[mask]
        break

    return filtered.reset_index(drop=True)


def compute_representative_stats(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    precision: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Compute representative statistics (mean/median/min/max) for specified columns."""

    if df is None or df.empty:
        return {}

    if columns is None:
        columns = df.select_dtypes(include=["number", "float", "int"]).columns

    stats: Dict[str, Dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        series = series.dropna()
        if series.empty:
            continue

        stats[column] = {
            "mean": round(float(series.mean()), precision),
            "median": round(float(series.median()), precision),
            "min": round(float(series.min()), precision),
            "max": round(float(series.max()), precision),
        }

    return stats


def encode_file_to_base64(path: Path, max_bytes: int = 512_000) -> Optional[str]:
    """Encode a binary file to Base64 string with optional size guard."""

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    if file_path.stat().st_size > max_bytes:
        logger.warning(
            "File too large for base64 encoding (size=%s bytes, limit=%s): %s",
            file_path.stat().st_size,
            max_bytes,
            file_path,
        )
        return None

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded
