"""
File operation utilities
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

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
