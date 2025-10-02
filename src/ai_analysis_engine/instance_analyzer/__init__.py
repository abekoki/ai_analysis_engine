"""
AI分析エンジンライブラリ - 時系列データ分析の自動化プラットフォーム

上位システム統合向けのシンプルで使いやすいAPIを提供
"""

from .library_api import AIAnalysisEngine
from .instance_analyzer import InstanceAnalyzer  # noqa: F401
from .config.library_config import AnalysisConfig
from .models.result import AnalysisResult, Hypothesis, AnalysisMetrics
from .exceptions import (
    AIAnalysisError,
    ConfigurationError,
    ValidationError,
    AnalysisError,
    TimeoutError,
    ResourceError,
    InitializationError
)
from .utils.context_recorder import context_recorder, AgentInteractionLogger

# 後方互換性のためのインポート
from .main import AIAnalysisEngine as _InternalEngine
from .config import config as _internal_config

__version__ = "1.0.0"
__author__ = "AI Analysis Engine Team"

__all__ = [
    # メインAPI
    "AIAnalysisEngine",
    "AnalysisConfig",
    "AnalysisResult",
    "Hypothesis",
    "AnalysisMetrics",

    # 例外クラス
    "AIAnalysisError",
    "ConfigurationError",
    "ValidationError",
    "AnalysisError",
    "TimeoutError",
    "ResourceError",
    "InitializationError",

    # 旧API互換
    "InstanceAnalyzer",

    # 後方互換性
    "_InternalEngine",
    "_internal_config",
    "context_recorder",
    "AgentInteractionLogger",
]
