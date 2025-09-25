"""
ライブラリ設定クラス - 上位システム統合向けのシンプルな設定管理
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AnalysisConfig(BaseModel):
    """
    分析エンジンの設定クラス

    上位システムからの利用を考慮したシンプルな設定インターフェースを提供
    """

    # OpenAI設定
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI APIキー"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="使用するOpenAIモデル"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="生成温度（0.0-2.0）"
    )
    max_tokens: int = Field(
        default=4000,
        gt=0,
        description="最大トークン数"
    )

    # システム設定
    timeout: int = Field(
        default=300,
        gt=0,
        description="分析タイムアウト（秒）"
    )
    output_dir: str = Field(
        default="./analysis_results",
        description="出力ディレクトリ"
    )
    log_level: str = Field(
        default="INFO",
        description="ログレベル"
    )

    # 高度な設定（オプション）
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="最大反復回数"
    )
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="ベクトルストアチャンクサイズ"
    )
    enable_plots: bool = Field(
        default=True,
        description="プロット生成を有効化"
    )

    class Config:
        """Pydantic設定"""
        validate_assignment = True
        extra = 'allow'  # 追加フィールドを許可

    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        """
        環境変数から設定を作成

        Returns:
            AnalysisConfig: 環境変数に基づく設定
        """
        return cls(
            api_key=os.getenv("AI_ANALYSIS_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            model=os.getenv("AI_ANALYSIS_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("AI_ANALYSIS_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("AI_ANALYSIS_MAX_TOKENS", "4000")),
            timeout=int(os.getenv("AI_ANALYSIS_TIMEOUT", "300")),
            output_dir=os.getenv("AI_ANALYSIS_OUTPUT_DIR", "./analysis_results"),
            log_level=os.getenv("AI_ANALYSIS_LOG_LEVEL", "INFO"),
            max_iterations=int(os.getenv("AI_ANALYSIS_MAX_ITERATIONS", "10")),
            chunk_size=int(os.getenv("AI_ANALYSIS_CHUNK_SIZE", "1000")),
            enable_plots=os.getenv("AI_ANALYSIS_ENABLE_PLOTS", "true").lower() == "true"
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        """
        辞書から設定を作成

        Args:
            config_dict: 設定辞書

        Returns:
            AnalysisConfig: 辞書に基づく設定
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書に変換

        Returns:
            Dict[str, Any]: 設定辞書
        """
        return self.model_dump()

    def validate(self) -> bool:
        """
        設定の妥当性を検証

        Returns:
            bool: 設定が妥当な場合True
        """
        if not self.api_key.strip():
            raise ValueError("OpenAI API key is required")

        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")

        return True

    def update(self, **kwargs) -> None:
        """
        設定を更新

        Args:
            **kwargs: 更新する設定値
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 設定の検証
        self.validate()
