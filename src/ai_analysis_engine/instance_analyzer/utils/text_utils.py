"""
Text processing utilities
"""

import json
from typing import Dict, Any, Optional, List

from .logger import get_logger
from .exploration_utils import (
    extract_json_with_llm,
    extract_frame_range_with_llm,
    extract_columns_with_llm,
    parse_condition_with_llm
)

logger = get_logger(__name__)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text response using LLM

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
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
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
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove non-printable characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

    return text


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap for better context retention

    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If not the last chunk, try to find a good breaking point
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            last_period = text.rfind('.', end - 100, end)
            last_newline = text.rfind('\n', end - 100, end)
            last_space = text.rfind(' ', end - 50, end)

            # Use the latest good breaking point
            break_point = max(last_period, last_newline, last_space)
            if break_point > start + chunk_size // 2:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = max(end - overlap, start + 1)

        # Prevent infinite loop
        if start >= len(text):
            break

    return chunks


def extract_frame_range(text: str) -> Optional[tuple]:
    """
    Extract frame range from natural language text using LLM

    Args:
        text: Natural language text describing frame range

    Returns:
        Tuple of (start_frame, end_frame) or None
    """
    try:
        # Use LLM-based extraction only
        result = extract_frame_range_with_llm(text)
        if result is not None:
            logger.info(f"Successfully extracted frame range using LLM: {result}")
            return result

        return None
    except Exception as e:
        raise RuntimeError(f"LLM-based frame range extraction failed: {e}") from e


def extract_column_info(text: str) -> List[str]:
    """
    Extract column names mentioned in text using LLM

    Args:
        text: Text mentioning column names

    Returns:
        List of column names found
    """
    try:
        # Use LLM-based extraction only
        columns = extract_columns_with_llm(text)
        if columns:
            logger.info(f"Successfully extracted columns using LLM: {columns}")
            return columns

        return []
    except Exception as e:
        raise RuntimeError(f"LLM-based column extraction failed: {e}") from e


def extract_expected_value(text: str) -> Optional[str]:
    """
    Extract expected value from natural language description using LLM

    Args:
        text: Natural language description

    Returns:
        Expected value description
    """
    try:
        # Use LLM-based condition parsing only
        condition = parse_condition_with_llm(text)
        if condition and condition.get('condition'):
            logger.info(f"Successfully extracted expected value using LLM: {condition['condition']}")
            return clean_text(condition['condition'])

        return None
    except Exception as e:
        raise RuntimeError(f"LLM-based expected value extraction failed: {e}") from e


def parse_boolean_condition(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse boolean conditions from natural language using LLM

    Args:
        text: Natural language condition

    Returns:
        Dictionary with parsed condition or None
    """
    try:
        # Use LLM-based condition parsing only
        condition = parse_condition_with_llm(text)
        if condition:
            logger.info(f"Successfully parsed boolean condition using LLM: {condition}")
            return condition

        return None
    except Exception as e:
        raise RuntimeError(f"LLM-based boolean condition parsing failed: {e}") from e


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity score

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Simple word overlap similarity
    words1 = set(clean_text(text1).lower().split())
    words2 = set(clean_text(text2).lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)
