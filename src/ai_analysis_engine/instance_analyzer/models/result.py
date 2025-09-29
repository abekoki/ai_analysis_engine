"""
ライブラリ結果モデル - 上位システム統合向けの構造化結果
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """仮説情報"""
    text: str = Field(..., description="仮説の内容")
    confidence: float = Field(..., ge=0.0, le=1.0, description="信頼度（0.0-1.0）")
    evidence: List[str] = Field(default_factory=list, description="根拠")
    category: str = Field(default="", description="カテゴリ")


class AnalysisMetrics(BaseModel):
    """分析メトリクス"""
    execution_time: float = Field(default=0.0, ge=0.0, description="実行時間（秒）")
    data_points_processed: int = Field(default=0, ge=0, description="処理されたデータポイント数")
    hypotheses_generated: int = Field(default=0, ge=0, description="生成された仮説数")
    plots_generated: int = Field(default=0, ge=0, description="生成されたプロット数")


class AnalysisResult(BaseModel):
    """
    分析結果クラス

    上位システムが扱いやすい構造化された結果を提供
    """

    # 基本情報
    success: bool = Field(..., description="分析の成功フラグ")
    dataset_id: str = Field(..., description="データセットID")

    # 結果内容
    report: Optional[str] = Field(default=None, description="生成されたレポート（Markdown）")
    summary: Optional[str] = Field(default=None, description="分析結果の簡潔な要約")

    # 生成物
    plots: List[str] = Field(default_factory=list, description="生成されたプロットファイルのパス")
    report_path: Optional[str] = Field(default=None, description="レポートファイルのパス")

    # 分析結果
    hypotheses: List[Hypothesis] = Field(default_factory=list, description="生成された仮説")
    metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics, description="分析メトリクス")

    # メタデータ
    timestamp: datetime = Field(default_factory=datetime.now, description="分析実行時刻")
    engine_version: str = Field(default="1.0.0", description="エンジンバージョン")

    # エラー情報
    error: Optional[str] = Field(default=None, description="エラーメッセージ")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="詳細なエラー情報")

    # 生データ（デバッグ用）
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="生の分析データ")

    class Config:
        """Pydantic設定"""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def success_result(
        cls,
        dataset_id: str,
        report: str,
        hypotheses: List[Hypothesis],
        plots: List[str] = None,
        summary: str = None,
        metrics: AnalysisMetrics = None,
        report_path: str = None,
        raw_data: Dict[str, Any] | None = None,
    ) -> "AnalysisResult":
        """
        成功時の結果を作成

        Args:
            dataset_id: データセットID
            report: 生成されたレポート
            hypotheses: 生成された仮説
            plots: 生成されたプロット
            summary: 要約
            metrics: メトリクス
            report_path: レポートファイルパス

        Returns:
            AnalysisResult: 成功結果
        """
        return cls(
            success=True,
            dataset_id=dataset_id,
            report=report,
            summary=summary,
            hypotheses=hypotheses,
            plots=plots or [],
            metrics=metrics or AnalysisMetrics(),
            report_path=report_path,
            raw_data=raw_data,
        )

    @classmethod
    def error_result(
        cls,
        dataset_id: str,
        error: str,
        error_details: Dict[str, Any] = None
    ) -> "AnalysisResult":
        """
        エラー時の結果を作成

        Args:
            dataset_id: データセットID
            error: エラーメッセージ
            error_details: 詳細なエラー情報

        Returns:
            AnalysisResult: エラー結果
        """
        return cls(
            success=False,
            dataset_id=dataset_id,
            error=error,
            error_details=error_details,
            metrics=AnalysisMetrics()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        結果を辞書に変換（JSONシリアライズ用）

        Returns:
            Dict[str, Any]: 辞書形式の結果
        """
        data = self.model_dump()

        # datetimeを文字列に変換
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()

        # Hypothesisを辞書に変換
        if 'hypotheses' in data and data['hypotheses']:
            if isinstance(data['hypotheses'][0], dict):
                # 既に辞書の場合はそのまま
                pass
            else:
                # Pydanticオブジェクトの場合は変換
                data['hypotheses'] = [h.model_dump() if hasattr(h, 'model_dump') else h for h in data['hypotheses']]

        # AnalysisMetricsを辞書に変換
        if 'metrics' in data and data['metrics']:
            if isinstance(data['metrics'], dict):
                # 既に辞書の場合はそのまま
                pass
            else:
                # Pydanticオブジェクトの場合は変換
                data['metrics'] = data['metrics'].model_dump() if hasattr(data['metrics'], 'model_dump') else data['metrics']

        return data

    def to_json(self) -> str:
        """
        結果をJSON文字列に変換

        Returns:
            str: JSON形式の結果
        """
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_to_file(self, file_path: str) -> None:
        """
        結果をファイルに保存

        Args:
            file_path: 保存先ファイルパス
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """
        辞書から結果を作成

        Args:
            data: 結果データ

        Returns:
            AnalysisResult: 復元された結果
        """
        # timestampをdatetimeに変換
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # hypothesesをHypothesisオブジェクトに変換
        if 'hypotheses' in data:
            data['hypotheses'] = [Hypothesis(**h) for h in data['hypotheses']]

        # metricsをAnalysisMetricsオブジェクトに変換
        if 'metrics' in data:
            data['metrics'] = AnalysisMetrics(**data['metrics'])

        return cls(**data)

    @classmethod
    def load_from_file(cls, file_path: str) -> "AnalysisResult":
        """
        ファイルから結果を読み込み

        Args:
            file_path: 読み込み元ファイルパス

        Returns:
            AnalysisResult: 読み込まれた結果
        """
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
