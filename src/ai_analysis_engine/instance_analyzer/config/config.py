"""
Configuration management for the AI Analysis Engine
Generic configuration system supporting multiple algorithms
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

from ..utils.exploration_utils import (
    extract_columns_with_llm,
    extract_thresholds_with_llm,
    extract_json_with_llm
)


class OpenAIConfig(BaseModel):
    """OpenAI API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4000)


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    persist_directory: str = Field(default="./data/vectorstore")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)


class REPLConfig(BaseModel):
    """REPL tool configuration"""
    timeout: int = Field(default=30)
    max_output_length: int = Field(default=5000)
    allowed_modules: list = Field(default_factory=lambda: ["pandas", "numpy", "matplotlib"])


class LangGraphConfig(BaseModel):
    """LangGraph configuration"""
    max_iterations: int = Field(default=10)
    checkpoint_path: str = Field(default="./data/checkpoints")


class AlgorithmConfig(BaseModel):
    """Algorithm-specific configuration parsed from specification documents"""

    # Basic algorithm information
    name: str = Field(default="")
    description: str = Field(default="")

    # Input/Output specifications
    input_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)

    # Thresholds and parameters (dynamic from spec)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Data validation rules
    value_ranges: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    valid_values: Dict[str, List[Any]] = Field(default_factory=dict)

    # Analysis patterns for detection
    detection_patterns: Dict[str, Any] = Field(default_factory=dict)

    # Evaluation criteria
    evaluation_criteria: Dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    """Analysis workflow configuration based on detailed analysis procedure"""

    # Analysis phases (from 解析手順詳細.md)
    phases: List[str] = Field(default_factory=lambda: [
        "analysis_preparation",
        "undetected_data_extraction",
        "undetected_data_characteristics",
        "algorithm_behavior_analysis",
        "cause_identification",
        "improvement_proposals",
        "cause_report_creation"
    ])

    # Data processing settings
    processing_rules: Dict[str, Any] = Field(default_factory=dict)

    # Pattern recognition settings
    pattern_analysis: Dict[str, Any] = Field(default_factory=dict)

    # Reporting templates
    report_templates: Dict[str, str] = Field(default_factory=dict)


class Config:
    """Main configuration class"""

    def __init__(self):
        self.openai = OpenAIConfig()
        self.vectorstore = VectorStoreConfig()
        self.repl = REPLConfig()
        self.langgraph = LangGraphConfig()
        self.algorithm = AlgorithmConfig()
        self.analysis = AnalysisConfig()

        # Project paths
        self.project_root = Path(".")
        self.data_dir = Path("./data")
        self.output_dir = Path("./outputs")
        self.logs_dir = Path("./logs")
        self.config_dir = Path("./config")

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        return cls()

    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            self.output_dir,
            self.logs_dir,
            Path(self.vectorstore.persist_directory),
            Path(self.langgraph.checkpoint_path)
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_vectorstore_path(self, segment: str) -> str:
        """Get vectorstore path for specific segment"""
        return str(Path(self.vectorstore.persist_directory) / f"{segment}_vectorstore")

    def validate_api_keys(self) -> bool:
        """Validate required API keys"""
        return bool(self.openai.api_key.strip())

    def load_algorithm_config_from_spec(self, spec_content: str) -> AlgorithmConfig:
        """
        Parse algorithm specification and create configuration using LLM-based extraction
        Based on detailed analysis procedure (解析手順詳細.md)

        Args:
            spec_content: Algorithm specification document content

        Returns:
            AlgorithmConfig: Parsed algorithm configuration
        """
        config = AlgorithmConfig()

        try:
            # Extract algorithm name and description using regex (keep this simple)
            import re
            name_match = re.search(r'#+\s*([^\n]+)', spec_content)
            if name_match:
                config.name = name_match.group(1).strip()

            # Extract thresholds using LLM
            thresholds = extract_thresholds_with_llm(spec_content)
            config.thresholds.update(thresholds)

            # Extract column names using LLM
            columns = extract_columns_with_llm(spec_content)
            config.input_columns.extend(columns)

            # Extract output columns from output specification table using LLM
            output_section = ""
            output_match = re.search(r'### 2\.2 出力仕様.*?(?=###|$)', spec_content, re.DOTALL)
            if output_match:
                output_section = output_match.group(0)

            if output_section:
                output_columns = extract_columns_with_llm(output_section)
                config.output_columns.extend(output_columns)

            # Set default required columns if not found
            if not config.required_columns:
                config.required_columns = ['frame_num', 'timestamp']  # Common defaults

            # Extract value ranges (keep simple regex for ranges)
            range_patterns = [
                r'([0-9.]+)\s*[-~]\s*([0-9.]+)',  # Range pattern
                r'範囲[:\s]*([0-9.]+)\s*[-~]\s*([0-9.]+)',  # Japanese range
            ]

            for pattern in range_patterns:
                matches = re.findall(pattern, spec_content)
                for i, match in enumerate(matches):
                    min_val, max_val = map(float, match)
                    config.value_ranges[f"range_{i}"] = {"min": min_val, "max": max_val}

            # Set default value ranges for input columns if not found
            for col in config.input_columns:
                if col not in [k for ranges in config.value_ranges.values() for k in ranges.keys()]:
                    # Extract default ranges from specification if available
                    if 'openness' in col.lower():
                        config.value_ranges[col] = {"min": 0.0, "max": 1.0}
                    elif 'confidence' in col.lower():
                        config.value_ranges[col] = {"min": 0.0, "max": 1.0}
                    elif 'probability' in col.lower():
                        config.value_ranges[col] = {"min": 0.0, "max": 1.0}

            # Extract valid values for categorical outputs (keep simple regex)
            for col in config.output_columns:
                if col in spec_content:
                    # Look for valid values in specification (e.g., 0/1, -1/0/1, etc.)
                    if 'detection' in col.lower() or 'result' in col.lower():
                        # Try to extract from specification patterns
                        valid_value_patterns = [
                            r'値[:\s]*([-1,0-9\s]+)',  # Japanese values
                            r'values[:\s]*([-1,0-9\s]+)',  # English values
                            r'[-1,0-9\s]+\([^)]*\)'  # Pattern like -1(Error), 0(Normal), 1(Drowsy)
                        ]
                        for pattern in valid_value_patterns:
                            matches = re.findall(pattern, spec_content, re.IGNORECASE)
                            if matches:
                                # Parse valid values from matches
                                for match in matches:
                                    values = [int(v.strip()) for v in match.split(',') if v.strip().lstrip('-').isdigit()]
                                    if values and col not in config.valid_values:
                                        config.valid_values[col] = sorted(list(set(values)))

            # Extract detection patterns using LLM
            from ..utils.exploration_utils import exploration_tool
            detection_patterns = exploration_tool.extract_patterns_from_spec(spec_content)
            config.detection_patterns.update(detection_patterns)

            # Add threshold-based patterns
            for threshold_name, threshold_value in config.thresholds.items():
                if 'time' in threshold_name.lower() or 'duration' in threshold_name.lower():
                    pattern_name = threshold_name.lower().replace('_', ' ')
                    config.detection_patterns[pattern_name] = {
                        "threshold": threshold_value,
                        "type": "time_based"
                    }
                elif 'threshold' in threshold_name.lower():
                    pattern_name = threshold_name.lower().replace('_threshold', '').replace('_', ' ')
                    config.detection_patterns[pattern_name] = {
                        "threshold": threshold_value,
                        "type": "value_based"
                    }

            # Set evaluation criteria
            config.evaluation_criteria = {
                "detection_accuracy": "Compare algorithm output with expected detection intervals",
                "false_positive_rate": "Calculate false detection rate in non-target intervals",
                "temporal_precision": "Evaluate detection timing accuracy"
            }

        except Exception as e:
            # LLM extraction failed - raise error instead of fallback
            raise RuntimeError(f"Algorithm configuration extraction failed: {e}") from e

    def load_algorithm_config_from_file(self, spec_file_path: str) -> AlgorithmConfig:
        """
        Load algorithm configuration from specification file

        Args:
            spec_file_path: Path to algorithm specification markdown file

        Returns:
            AlgorithmConfig: Loaded configuration
        """
        try:
            with open(spec_file_path, 'r', encoding='utf-8') as f:
                spec_content = f.read()

            config = self.load_algorithm_config_from_spec(spec_content)
            self.algorithm = config
            return config

        except Exception as e:
            # Return default configuration if parsing fails
            default_config = AlgorithmConfig()
            default_config.name = "Unknown Algorithm"
            default_config.description = f"Failed to parse specification: {e}"
            return default_config

    def save_algorithm_config(self, config: AlgorithmConfig, output_path: str) -> None:
        """
        Save algorithm configuration to YAML file

        Args:
            config: AlgorithmConfig to save
            output_path: Output file path
        """
        config_dict = config.model_dump()

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)

    def load_algorithm_config_from_yaml(self, yaml_path: str) -> AlgorithmConfig:
        """
        Load algorithm configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            AlgorithmConfig: Loaded configuration
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            config = AlgorithmConfig(**config_dict)
            self.algorithm = config
            return config

        except Exception as e:
            # Return default configuration if loading fails
            default_config = AlgorithmConfig()
            default_config.name = "Unknown Algorithm"
            default_config.description = f"Failed to load YAML config: {e}"
            return default_config

    def get_dynamic_analysis_config(self) -> AnalysisConfig:
        """
        Get analysis configuration based on loaded algorithm config
        Following the detailed analysis procedure (解析手順詳細.md)

        Returns:
            AnalysisConfig: Dynamic analysis configuration
        """
        analysis_config = AnalysisConfig()

        # Set processing rules based on algorithm config
        analysis_config.processing_rules = {
            "filter_criteria": {
                "evaluation_intervals": "Extract data within specified time ranges",
                "detection_flags": f"Filter by {self.algorithm.output_columns} values",
                "threshold_compliance": f"Check against {list(self.algorithm.thresholds.keys())}"
            },
            "feature_extraction": {
                "time_series": "Extract temporal patterns and trends",
                "statistical_measures": "Calculate mean, std, min, max for each feature",
                "correlation_analysis": "Analyze relationships between input and output features"
            }
        }

        # Set pattern analysis based on algorithm type
        if self.algorithm.detection_patterns:
            analysis_config.pattern_analysis = {
                "detection_patterns": self.algorithm.detection_patterns,
                "temporal_analysis": "Analyze detection timing and continuity",
                "feature_correlation": "Correlate input features with detection results"
            }

        # Set report templates
        analysis_config.report_templates = {
            "cause_analysis": """
## 原因分析レポート

### アルゴリズム仕様
- 名称: {algorithm_name}
- 閾値設定: {thresholds}
- 入力特徴量: {input_columns}

### 未検知データ特性
- データ範囲: {data_range}
- 特徴量分布: {feature_distribution}

### アルゴリズム挙動分析
- 検知ロジック: {detection_logic}
- 出力パターン: {output_patterns}

### 改善提案
{improvement_suggestions}
            """,
            "evaluation_summary": """
## 評価結果サマリー

### 全体評価
- 正解率: {accuracy}
- 誤検知率: {false_positive_rate}
- 未検知率: {false_negative_rate}

### 時系列分析
{temporal_analysis}

### 推奨改善策
{recommendations}
            """
        }

        self.analysis = analysis_config
        return analysis_config


# Global configuration instance
config = Config()
