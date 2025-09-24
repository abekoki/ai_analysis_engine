"""
カスタム例外クラス - ライブラリ利用者向けの明確なエラーハンドリング
"""

from typing import Dict, Any, Optional


class AIAnalysisError(Exception):
    """
    AI分析エンジンの基底例外クラス

    すべてのライブラリ例外の基底クラス
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """例外を辞書形式に変換"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(AIAnalysisError):
    """
    設定関連のエラー

    APIキー、モデル設定、環境設定などの問題
    """
    pass


class ValidationError(AIAnalysisError):
    """
    入力検証エラー

    ファイル形式、データ構造、パラメータ値などの問題
    """
    pass


class AnalysisError(AIAnalysisError):
    """
    分析実行時のエラー

    分析プロセス中の問題（LLMエラー、データ処理エラーなど）
    """
    pass


class TimeoutError(AIAnalysisError):
    """
    タイムアウトエラー

    分析が指定時間内に完了しなかった場合
    """
    pass


class ResourceError(AIAnalysisError):
    """
    リソース関連のエラー

    メモリ不足、ディスク容量不足、ネットワークエラーなど
    """
    pass


class InitializationError(AIAnalysisError):
    """
    初期化エラー

    エンジンの初期化に失敗した場合
    """
    pass
