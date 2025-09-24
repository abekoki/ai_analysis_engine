# AI分析エンジン ライブラリ仕様書

## 概要

汎用AI分析エンジンをライブラリ化し、上位システムからの統合利用を可能にするための仕様書です。

## 目的

- 上位システムからのシームレスな統合
- シンプルで直感的なAPI提供
- 設定の柔軟性と拡張性
- 堅牢なエラーハンドリング
- 包括的なテストカバレッジ

## アーキテクチャ

### コアコンポーネント

```python
from ai_analysis_engine import AIAnalysisEngine, AnalysisConfig, AnalysisResult

# 基本的な使用例
engine = AIAnalysisEngine()
config = AnalysisConfig(api_key="your-key", model="gpt-4o-mini")
result = engine.analyze(
    algorithm_output="path/to/algo.csv",
    core_output="path/to/core.csv",
    algorithm_spec="path/to/spec.md",
    expected_result="期待される動作の記述"
)
```

### クラス設計

#### AIAnalysisEngine (メインクラス)

**責任**: 分析エンジンのライフサイクル管理とAPI提供

**メソッド**:
- `__init__(config: AnalysisConfig = None)`: 初期化
- `initialize() -> bool`: エンジン初期化
- `analyze(...) -> AnalysisResult`: 分析実行
- `analyze_batch(...) -> List[AnalysisResult]`: バッチ分析
- `shutdown()`: クリーンアップ

#### AnalysisConfig (設定クラス)

**責任**: エンジン動作の設定管理

**属性**:
- `api_key: str`: OpenAI APIキー
- `model: str`: 使用するモデル
- `temperature: float`: 生成温度
- `timeout: int`: タイムアウト時間
- `output_dir: str`: 出力ディレクトリ

#### AnalysisResult (結果クラス)

**責任**: 分析結果の構造化表現

**属性**:
- `success: bool`: 実行成功フラグ
- `dataset_id: str`: データセットID
- `report: str`: 生成されたレポート
- `plots: List[str]`: 生成されたプロットパス
- `hypotheses: List[Hypothesis]`: 生成された仮説
- `execution_time: float`: 実行時間
- `error: Optional[str]`: エラーメッセージ

## API仕様

### 初期化API

```python
# デフォルト設定での初期化
engine = AIAnalysisEngine()

# カスタム設定での初期化
config = AnalysisConfig(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.1,
    timeout=300,
    output_dir="./results"
)
engine = AIAnalysisEngine(config)
```

### 単一分析API

```python
result = engine.analyze(
    algorithm_output="data/algo_output.csv",    # 必須
    core_output="data/core_output.csv",         # 必須
    algorithm_spec="docs/algo_spec.md",         # 必須
    expected_result="期待される動作記述",       # 必須
    algorithm_code=["src/algo.py"],             # オプション
    dataset_id="custom_dataset"                 # オプション
)
```

### バッチ分析API

```python
results = engine.analyze_batch([
    {
        "algorithm_output": "data/dataset1_algo.csv",
        "core_output": "data/dataset1_core.csv",
        "algorithm_spec": "docs/dataset1_spec.md",
        "expected_result": "データセット1の期待結果",
        "dataset_id": "dataset_1"
    },
    {
        "algorithm_output": "data/dataset2_algo.csv",
        "core_output": "data/dataset2_core.csv",
        "algorithm_spec": "docs/dataset2_spec.md",
        "expected_result": "データセット2の期待結果",
        "dataset_id": "dataset_2"
    }
])
```

### 同期/非同期API

```python
# 同期実行（デフォルト）
result = engine.analyze(...)

# 非同期実行
import asyncio
result = await engine.analyze_async(...)
```

## エラーハンドリング

### エラー分類

1. **ConfigurationError**: 設定関連エラー
2. **ValidationError**: 入力データ検証エラー
3. **AnalysisError**: 分析実行時エラー
4. **TimeoutError**: タイムアウトエラー

### エラー処理例

```python
try:
    result = engine.analyze(...)
    if not result.success:
        print(f"Analysis failed: {result.error}")
except AnalysisError as e:
    print(f"Analysis error: {e}")
    # 適切なエラーハンドリング
```

## 設定管理

### 環境変数

```bash
# 必須
export OPENAI_API_KEY="your-api-key"

# オプション
export AI_ANALYSIS_MODEL="gpt-4o-mini"
export AI_ANALYSIS_TIMEOUT="300"
export AI_ANALYSIS_OUTPUT_DIR="./results"
```

### プログラム内設定

```python
from ai_analysis_engine import AnalysisConfig

config = AnalysisConfig()
config.api_key = "your-key"
config.model = "gpt-4"
config.timeout = 600
```

## 依存関係

### 必須依存関係

- `langchain>=0.2.0`
- `langgraph>=0.1.0`
- `pandas`
- `matplotlib`
- `seaborn`
- `faiss-cpu`
- `pydantic>=2.0.0`

### オプション依存関係

- `openai` (OpenAI API使用時)
- `python-dotenv` (環境変数ファイル使用時)

## テスト仕様

### ユニットテスト

- 各コンポーネントの単体テスト
- モックを使用した外部依存関係のテスト
- エラーケースのテスト

### 統合テスト

- エンド-to-エンドの分析フロー
- 複数データセットのバッチ処理
- 設定変更時の動作確認

### パフォーマンステスト

- 大規模データの処理性能
- メモリ使用量の監視
- タイムアウト処理の検証

## パッケージング

### ディレクトリ構造

```
ai_analysis_engine/
├── __init__.py          # パッケージ初期化
├── core/               # コア機能
│   ├── __init__.py
│   ├── engine.py       # メインエンジン
│   └── analyzer.py     # 分析実行クラス
├── config/             # 設定管理
│   ├── __init__.py
│   └── config.py       # 設定クラス
├── models/             # データモデル
│   ├── __init__.py
│   ├── result.py       # 結果モデル
│   └── types.py        # 型定義
├── utils/              # ユーティリティ
│   ├── __init__.py
│   ├── validation.py   # 入力検証
│   └── formatting.py   # フォーマット処理
└── exceptions/         # カスタム例外
    ├── __init__.py
    └── errors.py       # エラークラス
```

### ビルド設定

```toml
# pyproject.toml
[project]
name = "ai-analysis-engine"
version = "1.0.0"
description = "汎用AI分析エンジンライブラリ"
dependencies = [
    "langchain>=0.2.0",
    "langgraph>=0.1.0",
    "pandas",
    "matplotlib",
    "seaborn",
    "faiss-cpu",
    "pydantic>=2.0.0"
]

[project.urls]
Homepage = "https://github.com/your-org/ai-analysis-engine"
Documentation = "https://ai-analysis-engine.readthedocs.io/"
```

## 使用例

### 基本的な使用例

```python
from ai_analysis_engine import AIAnalysisEngine

# エンジン初期化
engine = AIAnalysisEngine()

# 分析実行
result = engine.analyze(
    algorithm_output="data/algorithm_output.csv",
    core_output="data/core_output.csv",
    algorithm_spec="docs/algorithm_spec.md",
    expected_result="フレーム100-200の間に検知結果が存在すること"
)

# 結果確認
if result.success:
    print("Analysis completed successfully!")
    print(f"Report: {result.report}")
else:
    print(f"Analysis failed: {result.error}")
```

### 上位システム統合例

```python
from ai_analysis_engine import AIAnalysisEngine, AnalysisConfig
import os

class AnalysisService:
    def __init__(self):
        config = AnalysisConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            output_dir="./analysis_results"
        )
        self.engine = AIAnalysisEngine(config)

    def analyze_algorithm_performance(self, dataset_path: str) -> dict:
        """アルゴリズム性能分析サービス"""
        # データセット情報の構築
        algo_csv = f"{dataset_path}/algorithm_output.csv"
        core_csv = f"{dataset_path}/core_output.csv"
        spec_md = f"{dataset_path}/algorithm_spec.md"

        # 分析実行
        result = self.engine.analyze(
            algorithm_output=algo_csv,
            core_output=core_csv,
            algorithm_spec=spec_md,
            expected_result="期待される検知性能が発揮されていること"
        )

        return {
            "success": result.success,
            "report": result.report,
            "execution_time": result.execution_time,
            "hypotheses": [h.text for h in result.hypotheses]
        }
```

## 移行計画

### フェーズ1: API整理
- 現在の複雑なAPIをシンプルなインターフェースに整理
- 不要なパラメータの除去
- デフォルト値の最適化

### フェーズ2: 設定管理改善
- 環境変数とプログラム内設定の一体化
- 設定ファイル対応の追加
- 動的設定変更機能の実装

### フェーズ3: エラーハンドリング強化
- カスタム例外クラスの実装
- エラーメッセージの改善
- リカバリー機能の追加

### フェーズ4: テスト整備
- 包括的なテストスイートの作成
- CI/CDパイプラインの構築
- パフォーマンステストの追加

### フェーズ5: ドキュメント作成
- APIリファレンスの作成
- 使用例集の整備
- トラブルシューティングガイド作成
