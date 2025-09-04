"""設定管理モジュール

このモジュールは、YAML設定ファイルの読み込みと設定値の管理を行います。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Settings:
    """設定管理クラス"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Args:
            config_file: 設定ファイルのパス（相対パスまたは絶対パス）
        """
        if config_file is None:
            # デフォルトの設定ファイルパス
            config_file = Path(__file__).parent.parent.parent.parent / "config" / "default_config.yaml"

        self.config_file = Path(config_file)
        self._config: Dict[str, Any] = {}

        # 設定ファイルが存在しない場合はデフォルト設定を作成
        if not self.config_file.exists():
            self._create_default_config()

        self.load_config()

    def _create_default_config(self) -> None:
        """デフォルト設定ファイルを作成"""
        default_config = {
            "global": {
                "database_path": "../DataWareHouse/database.db",
                "datawarehouse_path": "../DataWareHouse/",
                "templates_path": "./templates/"
            },
            "orchestrator": {
                "max_parallel_instances": 4,
                "timeout_seconds": 900
            },
            "performance_analyzer": {
                "metrics": ["accuracy", "over_detection_count_per_hour"],
                "visualization_level": "standard"
            },
            "instance_analyzer": {
                "max_hypothesis_attempts": 3,
                "llm_model": "gpt-4",
                "temperature": 0.1,
                "only_problem_instances": True,
                "drowsy_detection": {
                    "left_eye_close_threshold": 0.10,
                    "right_eye_close_threshold": 0.10,
                    "continuous_close_time": 1.00,
                    "face_conf_threshold": 0.75
                }
            }
        }

        # 設定ディレクトリが存在しない場合は作成
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # YAMLファイルに書き出し
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    def load_config(self) -> None:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {self.config_file}") from e

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得（ドット区切りでネストしたキーを指定可能）

        Args:
            key: 設定キー（例: "global.database_path"）
            default: デフォルト値

        Returns:
            設定値
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        設定値を設定

        Args:
            key: 設定キー（ドット区切り）
            value: 設定値
        """
        keys = key.split('.')
        config = self._config

        # ネストした辞書を作成
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_config(self) -> None:
        """設定をファイルに保存"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    @property
    def config(self) -> Dict[str, Any]:
        """設定辞書全体を取得"""
        return self._config.copy()

    def __str__(self) -> str:
        """文字列表現"""
        return f"Settings(config_file={self.config_file})"
