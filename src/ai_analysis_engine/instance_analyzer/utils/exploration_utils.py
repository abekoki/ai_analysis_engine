"""
Natural Language Exploration Utilities
LLM-based pattern extraction and text analysis utilities
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# Lazy import to avoid circular import
# from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FrameInterval(BaseModel):
    """Pydantic model for extracting frame intervals from natural language text"""
    start_frame: Optional[int] = Field(None, description="Start frame number of the evaluation interval")
    end_frame: Optional[int] = Field(None, description="End frame number of the evaluation interval")
    has_interval: bool = Field(False, description="Whether a frame interval was found in the text")


class ColumnInfo(BaseModel):
    """Pydantic model for extracting column information from specifications"""
    columns: List[str] = Field(default_factory=list, description="List of column names found")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Column name to data type mapping")
    confidence_score: float = Field(0.0, description="Confidence score of the extraction (0.0 to 1.0)")


class ThresholdInfo(BaseModel):
    """Pydantic model for extracting threshold information from specifications"""
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Threshold name to value mapping")
    confidence_score: float = Field(0.0, description="Confidence score of the extraction (0.0 to 1.0)")


class PatternInfo(BaseModel):
    """Pydantic model for extracting pattern information from specifications"""
    patterns: Dict[str, Any] = Field(default_factory=dict, description="Pattern name to pattern data mapping")
    confidence_score: float = Field(0.0, description="Confidence score of the extraction (0.0 to 1.0)")


class ExplorationTool:
    """
    Natural language exploration tool using LLM
    Provides LLM-based alternatives to regex pattern matching
    """

    def __init__(self):
        # Lazy import to avoid circular import
        from ..config import config
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

    def extract_frame_interval(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract frame interval from natural language text using LLM

        Args:
            text: Natural language text containing frame interval information

        Returns:
            Tuple of (start_frame, end_frame) or (None, None) if not found
        """
        try:
            # Create prompt for frame interval extraction
            frame_extraction_prompt = f"""あなたはテキストからフレーム区間を抽出する専門家です。
以下のテキストから、評価対象となるフレーム区間を抽出してください。

テキスト内容:
{text}

以下のJSON形式で結果を返してください:
{{
    "start_frame": 開始フレーム番号（整数、見つからない場合はnull）,
    "end_frame": 終了フレーム番号（整数、見つからない場合はnull）,
    "has_interval": フレーム区間が見つかったかどうか（true/false）
}}

例:
{{"start_frame": 465, "end_frame": 593, "has_interval": true}}

テキストにフレーム区間が明記されていない場合は、start_frameとend_frameをnullにし、has_intervalをfalseにしてください。
フレーム区間は「フレーム区間XXX-YYY」や「XXX〜YYYフレーム」などの形式で表現されることが多いです。"""

            # Call LLM
            response = self.llm.invoke(frame_extraction_prompt)

            # Parse JSON response
            result = json.loads(response.content.strip())
            if result.get('has_interval') and result.get('start_frame') is not None and result.get('end_frame') is not None:
                start_frame = int(result['start_frame'])
                end_frame = int(result['end_frame'])
                logger.info(f"LLM extracted frame interval: {start_frame}-{end_frame}")
                return start_frame, end_frame
            else:
                logger.info("No frame interval found in text using LLM")
                return None, None

        except Exception as e:
            raise RuntimeError(f"LLM frame interval extraction failed: {e}") from e

    def extract_columns_from_spec(self, spec_content: str) -> List[str]:
        """
        Extract column information from specification content using LLM

        Args:
            spec_content: Specification content as string

        Returns:
            List of column names found

        Raises:
            RuntimeError: If LLM extraction fails
        """
        try:
            # Create prompt for column extraction
            column_extraction_prompt = f"""あなたはアルゴリズム仕様書から列名を抽出する専門家です。
以下の仕様書の内容から、使用されている列名を抽出してください。

仕様書内容:
{spec_content[:2000]}

以下のJSON形式で結果を返してください:
{{
    "columns": ["column_name1", "column_name2", "column_name3"],
    "data_types": {{"column_name1": "float", "column_name2": "int"}},
    "confidence_score": 0.85
}}

例:
{{"columns": ["frame_num", "leye_openness", "reye_openness", "confidence"], "data_types": {{"frame_num": "int", "leye_openness": "float"}}, "confidence_score": 0.9}}

列名は主に数値データを含む列に焦点を当ててください。"""

            # Call LLM
            response = self.llm.invoke(column_extraction_prompt)

            # Parse JSON response
            result = json.loads(response.content.strip())
            if result.get('columns') and len(result['columns']) > 0:
                columns = result['columns']
                logger.info(f"LLM extracted columns: {columns[:5]}... (confidence: {result.get('confidence_score', 0.0)})")
                return columns
            else:
                logger.info("No columns found in specification using LLM")
                return []

        except Exception as e:
            raise RuntimeError(f"LLM column extraction failed: {e}") from e

    def extract_thresholds_from_spec(self, spec_content: str) -> Dict[str, float]:
        """
        Extract threshold information from specification content using LLM

        Args:
            spec_content: Specification content as string

        Returns:
            Dictionary mapping threshold names to values

        Raises:
            RuntimeError: If LLM extraction fails
        """
        try:
            # Create prompt for threshold extraction
            threshold_extraction_prompt = f"""あなたはアルゴリズム仕様書から閾値を抽出する専門家です。
以下の仕様書の内容から、使用されている閾値情報を抽出してください。

仕様書内容:
{spec_content[:2000]}

以下のJSON形式で結果を返してください:
{{
    "thresholds": {{"threshold_name1": 0.5, "threshold_name2": 0.8}},
    "confidence_score": 0.85
}}

例:
{{"thresholds": {{"leye_openness_threshold": 0.3, "reye_openness_threshold": 0.3, "confidence_threshold": 0.7}}, "confidence_score": 0.9}}

閾値は主に数値の設定値に焦点を当ててください。"""

            # Call LLM
            response = self.llm.invoke(threshold_extraction_prompt)

            # Parse JSON response
            result = json.loads(response.content.strip())
            if result.get('thresholds') and len(result['thresholds']) > 0:
                thresholds = result['thresholds']
                logger.info(f"LLM extracted thresholds: {thresholds} (confidence: {result.get('confidence_score', 0.0)})")
                return thresholds
            else:
                logger.info("No thresholds found in specification using LLM")
                return {{}}

        except Exception as e:
            raise RuntimeError(f"LLM threshold extraction failed: {e}") from e

    def extract_patterns_from_spec(self, spec_content: str) -> Dict[str, Any]:
        """
        Extract pattern information from specification content using LLM

        Args:
            spec_content: Specification content as string

        Returns:
            Dictionary containing pattern information

        Raises:
            RuntimeError: If LLM extraction fails
        """
        try:
            # Create prompt for pattern extraction
            pattern_extraction_prompt = f"""あなたはアルゴリズム仕様書からパターン情報を抽出する専門家です。
以下の仕様書の内容から、検知パターンや処理パターンを抽出してください。

仕様書内容:
{spec_content[:2000]}

以下のJSON形式で結果を返してください:
{{
    "patterns": {{
        "pattern_name1": {{"type": "time_based", "threshold": 0.5}},
        "pattern_name2": {{"type": "value_based", "threshold": 0.8}}
    }},
    "confidence_score": 0.85
}}

例:
{{"patterns": {{"continuous_closure": {{"type": "time_based", "threshold": 3.0}}, "eye_openness": {{"type": "value_based", "threshold": 0.3}}}}, "confidence_score": 0.9}}

パターンは検知ロジックや処理ルールに関連するものに焦点を当ててください。"""

            # Call LLM
            response = self.llm.invoke(pattern_extraction_prompt)

            # Parse JSON response
            result = json.loads(response.content.strip())
            if result.get('patterns') and len(result['patterns']) > 0:
                patterns = result['patterns']
                logger.info(f"LLM extracted patterns: {patterns} (confidence: {result.get('confidence_score', 0.0)})")
                return patterns
            else:
                logger.info("No patterns found in specification using LLM")
                return {{}}

        except Exception as e:
            raise RuntimeError(f"LLM pattern extraction failed: {e}") from e

    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from natural language text using LLM

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON as dictionary, or None if not found
        """
        try:
            # Create prompt for JSON extraction
            json_extraction_prompt = f"""あなたはテキストからJSONを抽出する専門家です。
以下のテキストから、JSON形式のデータを抽出してください。

テキスト内容:
{text}

以下のJSON形式で結果を返してください:
{{
    "extracted_json": {{...}}  // テキストから抽出されたJSONデータ
}}

テキストにJSONが含まれていない場合は、extracted_jsonをnullにしてください。"""

            # Call LLM
            response = self.llm.invoke(json_extraction_prompt)

            # Parse JSON response
            result = json.loads(response.content.strip())
            if result.get('extracted_json') is not None:
                logger.info("LLM extracted JSON from text")
                return result['extracted_json']
            else:
                logger.info("No JSON found in text using LLM")
                return None

        except Exception as e:
            raise RuntimeError(f"LLM JSON extraction failed: {e}") from e

    def parse_boolean_condition(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse boolean condition from natural language text using LLM

        Args:
            text: Text containing boolean condition

        Returns:
            Dictionary with condition information, or None if not parseable
        """
        try:
            # Create prompt for boolean condition parsing
            condition_parsing_prompt = f"""あなたはテキストから条件式を解析する専門家です。
以下のテキストから、条件式や判定条件を解析してください。

テキスト内容:
{text}

以下のJSON形式で結果を返してください:
{{
    "condition": "条件式の文字列表現",
    "operator": "比較演算子（<, >, ==, etc.）",
    "threshold": 0.5,
    "field": "条件対象のフィールド名",
    "logic": "AND/ORなどの論理演算子"
}}

例:
{{"condition": "leye_openness < 0.3", "operator": "<", "threshold": 0.3, "field": "leye_openness", "logic": "AND"}}

条件が見つからない場合は、各フィールドをnullにしてください。"""

            # Call LLM
            response = self.llm.invoke(condition_parsing_prompt)

            # Parse JSON response
            result = json.loads(response.content.strip())
            if result.get('condition') is not None:
                logger.info(f"LLM parsed condition: {result['condition']}")
                return result
            else:
                logger.info("No condition found in text using LLM")
                return None

        except Exception as e:
            raise RuntimeError(f"LLM condition parsing failed: {e}") from e


# Lazy initialization of global exploration tool instance
_exploration_tool = None

def _get_exploration_tool():
    """Get global exploration tool instance with lazy initialization"""
    global _exploration_tool
    if _exploration_tool is None:
        _exploration_tool = ExplorationTool()
    return _exploration_tool


def extract_frame_range_with_llm(text: str) -> Optional[Tuple[int, int]]:
    """
    Extract frame range from text using LLM (convenience function)

    Args:
        text: Text to extract frame range from

    Returns:
        Tuple of (start_frame, end_frame) or None if not found
    """
    return _get_exploration_tool().extract_frame_interval(text)


def extract_columns_with_llm(spec_content: str) -> List[str]:
    """
    Extract columns from specification using LLM (convenience function)

    Args:
        spec_content: Specification content

    Returns:
        List of column names
    """
    return _get_exploration_tool().extract_columns_from_spec(spec_content)


def extract_thresholds_with_llm(spec_content: str) -> Dict[str, float]:
    """
    Extract thresholds from specification using LLM (convenience function)

    Args:
        spec_content: Specification content

    Returns:
        Dictionary of threshold names to values
    """
    return _get_exploration_tool().extract_thresholds_from_spec(spec_content)


def extract_json_with_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text using LLM (convenience function)

    Args:
        text: Text to extract JSON from

    Returns:
        Extracted JSON as dictionary or None
    """
    return _get_exploration_tool().extract_json_from_text(text)


def parse_condition_with_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse boolean condition from text using LLM (convenience function)

    Args:
        text: Text containing condition

    Returns:
        Dictionary with condition information or None
    """
    return _get_exploration_tool().parse_boolean_condition(text)
