# AI分析エンジンライブラリ使用ガイド

## 概要

AI分析エンジンライブラリは、時系列データ分析の自動化を目的としたPythonライブラリです。上位システムからの簡単な統合を可能にし、複雑な分析ワークフローをシンプルなAPIで利用できます。

## 🚀 クイックスタート

### 新しいAPI仕様での使用例

```python
from ai_analysis_engine import AIAnalysisEngine, AnalysisConfig

# 1. 設定の作成
config = AnalysisConfig(
    api_key="your-openai-api-key",
    model="gpt-4o-mini",
    timeout=300
)

# 2. エンジンの作成
engine = AIAnalysisEngine(config)

# 3. RAG初期化（必須）
if not engine.initialize(
    algorithm_specs=["docs/algorithm_spec.md"],
    algorithm_codes=["src/algorithm.py"],
    evaluation_specs=["docs/evaluation_spec.md"],
    evaluation_codes=["src/evaluation.py"]
):
    print("初期化失敗")
    exit(1)

# 4. 複数データセットの一括分析
results = engine.analyze(
    algorithm_outputs=[
        "data/dataset1_algorithm_output.csv",
        "data/dataset2_algorithm_output.csv"
    ],
    core_outputs=[
        "data/dataset1_core_output.csv",
        "data/dataset2_core_output.csv"
    ],
    expected_results=[
        "データセット1の期待結果",
        "データセット2の期待結果"
    ],
    output_dir="./analysis_results",
    dataset_ids=["dataset_1", "dataset_2"]
)

# 5. 結果の確認
for result in results:
    if result.success:
        print(f"✅ {result.dataset_id}: 分析成功")
        print(f"   レポート: {result.report[:100]}...")
        print(f"   仮説数: {len(result.hypotheses)}")
    else:
        print(f"❌ {result.dataset_id}: 分析失敗 - {result.error}")
```

### 環境変数を使用した設定

```bash
export OPENAI_API_KEY="your-api-key"
export AI_ANALYSIS_MODEL="gpt-4o-mini"
export AI_ANALYSIS_TIMEOUT="300"
```

```python
from ai_analysis_engine import AIAnalysisEngine, AnalysisConfig

# 環境変数から自動設定
config = AnalysisConfig.from_env()
engine = AIAnalysisEngine(config)
```

## APIリファレンス

### AIAnalysisEngine

メインの分析エンジンクラスです。

#### メソッド

##### `__init__(config=None)`
エンジンを初期化します。

- **パラメータ**:
  - `config` (AnalysisConfig, optional): 分析設定。Noneの場合はデフォルト設定を使用。

##### `initialize(algorithm_specs, algorithm_codes, evaluation_specs, evaluation_codes)`
エンジンの初期化とRAGベクトル化を実行します。

- **パラメータ**:
  - `algorithm_specs` (List[str]): アルゴリズム仕様Markdownファイルパスのリスト
  - `algorithm_codes` (List[str]): アルゴリズム実装コードファイルパスのリスト
  - `evaluation_specs` (List[str]): 評価仕様Markdownファイルパスのリスト
  - `evaluation_codes` (List[str]): 評価環境コードファイルパスのリスト

- **戻り値**: bool - 初期化成功の場合はTrue
- **例外**: InitializationError, ValidationError

##### `analyze(algorithm_outputs, core_outputs, expected_results, output_dir, dataset_ids, timeout=None)`
複数データセットの一括分析を実行します。

- **パラメータ**:
  - `algorithm_outputs` (List[str]): アルゴリズム出力CSVファイルパスのリスト
  - `core_outputs` (List[str]): コアライブラリ出力CSVファイルパスのリスト
  - `expected_results` (List[str]): 期待される結果の自然言語記述のリスト
  - `output_dir` (str): 結果出力ディレクトリ
  - `dataset_ids` (List[str]): データセットIDのリスト
  - `timeout` (int, optional): タイムアウト時間（秒）

- **戻り値**: List[AnalysisResult] - 分析結果のリスト
- **例外**: ValidationError, AnalysisError, TimeoutError

##### `analyze_async(...)`
非同期で複数データセットの一括分析を実行します。

- **パラメータ**: `analyze()`と同じ
- **戻り値**: Awaitable[List[AnalysisResult]]

##### `get_status() -> Dict`
エンジンの現在の状態を取得します。

- **戻り値**: 状態情報を含む辞書

##### `shutdown()`
エンジンをシャットダウンし、リソースを解放します。

### AnalysisConfig

分析設定を管理するクラスです。

#### 属性

- `api_key` (str): OpenAI APIキー
- `model` (str): 使用するモデル（デフォルト: "gpt-4o-mini"）
- `temperature` (float): 生成温度（0.0-2.0、デフォルト: 0.1）
- `max_tokens` (int): 最大トークン数（デフォルト: 4000）
- `timeout` (int): タイムアウト時間（秒、デフォルト: 300）
- `output_dir` (str): 出力ディレクトリ（デフォルト: "./analysis_results"）

#### クラスメソッド

##### `from_env() -> AnalysisConfig`
環境変数から設定を作成します。

##### `from_dict(config_dict) -> AnalysisConfig`
辞書から設定を作成します。

#### メソッド

##### `validate() -> bool`
設定の妥当性を検証します。

##### `to_dict() -> Dict`
設定を辞書に変換します。

##### `update(**kwargs)`
設定を更新します。

### AnalysisResult

分析結果を表すクラスです。

#### 属性

- `success` (bool): 分析の成功フラグ
- `dataset_id` (str): データセットID
- `report` (str, optional): 生成されたレポート
- `summary` (str, optional): 分析結果の要約
- `plots` (List[str]): 生成されたプロットファイルのパス
- `hypotheses` (List[Hypothesis]): 生成された仮説
- `metrics` (AnalysisMetrics): 分析メトリクス

#### クラスメソッド

##### `success_result(...) -> AnalysisResult`
成功時の結果を作成します。

##### `error_result(...) -> AnalysisResult`
エラー時の結果を作成します。

#### メソッド

##### `to_dict() -> Dict`
結果を辞書に変換します。

##### `to_json() -> str`
結果をJSON文字列に変換します。

##### `save_to_file(file_path)`
結果をファイルに保存します。

##### `load_from_file(file_path) -> AnalysisResult`
ファイルから結果を読み込みます。

## 詳細な使用例

### 1. 完全なワークフロー

```python
from ai_analysis_engine import AIAnalysisEngine, AnalysisConfig
import os

# APIキーの設定
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 設定の作成
config = AnalysisConfig(
    model="gpt-4o-mini",
    temperature=0.1,
    timeout=600
)

# エンジンの初期化
engine = AIAnalysisEngine(config)

# RAG初期化
success = engine.initialize(
    algorithm_specs=[
        "docs/algorithm_spec_v1.md",
        "docs/algorithm_spec_v2.md"
    ],
    algorithm_codes=[
        "src/algorithm/detector.py",
        "src/algorithm/utils.py"
    ],
    evaluation_specs=[
        "docs/evaluation_spec.md"
    ],
    evaluation_codes=[
        "src/evaluation/metrics.py"
    ]
)

if not success:
    print("RAG初期化失敗")
    exit(1)

print("✅ RAG初期化完了")

# 複数データセットの分析
results = engine.analyze(
    algorithm_outputs=[
        "data/test1_algorithm.csv",
        "data/test2_algorithm.csv",
        "data/test3_algorithm.csv"
    ],
    core_outputs=[
        "data/test1_core.csv",
        "data/test2_core.csv",
        "data/test3_core.csv"
    ],
    expected_results=[
        "テストケース1: 正常検知が期待される",
        "テストケース2: 誤検知の確認",
        "テストケース3: 境界条件のテスト"
    ],
    output_dir="./batch_analysis_results",
    dataset_ids=[
        "test_case_1",
        "test_case_2",
        "test_case_3"
    ],
    timeout=900  # 15分タイムアウト
)

# 結果の集計
successful = sum(1 for r in results if r.success)
total = len(results)

print(f"\\n📊 分析結果: {successful}/{total} 成功")

for result in results:
    status = "✅" if result.success else "❌"
    print(f"{status} {result.dataset_id}: {len(result.hypotheses)}仮説生成")

# エンジンのシャットダウン
engine.shutdown()
```

### 2. エラーハンドリング

```python
from ai_analysis_engine import (
    AIAnalysisError,
    ConfigurationError,
    ValidationError,
    AnalysisError,
    TimeoutError
)

try:
    # 初期化
    engine = AIAnalysisEngine()
    engine.initialize(
        algorithm_specs=["nonexistent.md"],  # 存在しないファイル
        algorithm_codes=["src/code.py"],
        evaluation_specs=["docs/eval.md"],
        evaluation_codes=["src/eval.py"]
    )

except ValidationError as e:
    print(f"入力検証エラー: {e}")
except InitializationError as e:
    print(f"初期化エラー: {e}")

try:
    # 分析実行
    results = engine.analyze(
        algorithm_outputs=["data/algo.csv"],
        core_outputs=["data/core.csv"],
        expected_results=["期待結果"],
        output_dir="",
        dataset_ids=["test"]
    )

except AnalysisError as e:
    print(f"分析エラー: {e}")
except TimeoutError as e:
    print(f"タイムアウト: {e}")
```

### 3. 非同期実行

```python
import asyncio

async def analyze_async():
    results = await engine.analyze_async(
        algorithm_outputs=["data/algo.csv"],
        core_outputs=["data/core.csv"],
        expected_results=["期待結果"],
        output_dir="./results",
        dataset_ids=["async_test"]
    )
    return results

# 非同期実行
results = asyncio.run(analyze_async())
```

### 4. エンジン状態の監視

```python
# 初期化前の状態
status = engine.get_status()
print(f"初期化済み: {status['initialized']}")
print(f"RAG初期化済み: {status['rag_initialized']}")

# 初期化後
engine.initialize([...])  # RAG初期化
status = engine.get_status()
print(f"読み込まれたドキュメント: {status['documents_loaded']}")

# 分析後
results = engine.analyze([...])
print(f"処理済みデータセット数: {len(results)}")
```

## 設定の詳細

### 環境変数

| 変数名 | 説明 | デフォルト値 |
|--------|------|-------------|
| `OPENAI_API_KEY` | OpenAI APIキー | 必須 |
| `AI_ANALYSIS_MODEL` | 使用モデル | gpt-4o-mini |
| `AI_ANALYSIS_TEMPERATURE` | 生成温度 | 0.1 |
| `AI_ANALYSIS_MAX_TOKENS` | 最大トークン数 | 4000 |
| `AI_ANALYSIS_TIMEOUT` | タイムアウト（秒） | 300 |
| `AI_ANALYSIS_OUTPUT_DIR` | 出力ディレクトリ | ./analysis_results |

### プログラム内設定

```python
from ai_analysis_engine import AnalysisConfig

config = AnalysisConfig(
    api_key="your-key",
    model="gpt-4",
    temperature=0.2,
    timeout=600,
    output_dir="/custom/output"
)
```

## パフォーマンスと制限

### パフォーマンス目安

- **小規模データ**（1,000行）: 30-60秒/データセット
- **中規模データ**（10,000行）: 2-5分/データセット
- **大規模データ**（100,000行）: 10-30分/データセット

### 制限事項

- OpenAI APIキーが必要
- インターネット接続が必要
- RAG初期化時に十分なメモリが必要
- 大規模データの場合はバッチサイズを考慮

## トラブルシューティング

### よくある問題

#### 1. RAG初期化エラー
```
ValidationError: algorithm_specs cannot be empty
```
**解決方法**: 初期化時に全ての必須パラメータを指定
```python
engine.initialize(
    algorithm_specs=["docs/spec.md"],  # 必須
    algorithm_codes=["src/code.py"],   # 必須
    evaluation_specs=["docs/eval.md"], # 必須
    evaluation_codes=["src/eval.py"]   # 必須
)
```

#### 2. 分析入力エラー
```
ValidationError: All input lists must have the same length
```
**解決方法**: 全ての入力リストの長さを一致させる
```python
results = engine.analyze(
    algorithm_outputs=["data1.csv", "data2.csv"],     # 長さ2
    core_outputs=["core1.csv", "core2.csv"],          # 長さ2
    expected_results=["期待1", "期待2"],              # 長さ2
    dataset_ids=["id1", "id2"]                        # 長さ2
)
```

#### 3. タイムアウト
```
TimeoutError: Batch analysis timed out after 300 seconds
```
**解決方法**: timeoutパラメータを調整
```python
results = engine.analyze(
    ..., timeout=900  # 15分に延長
)
```

## 移植時の注意事項

### 必須ファイル

プロジェクトにコピーする必要があるファイル：

```
src/ai_analysis_engine/
├── __init__.py
├── library_api.py
├── config/
│   ├── __init__.py
│   └── library_config.py
├── models/
│   ├── __init__.py
│   └── result.py
├── exceptions.py
├── main.py
├── agents/
├── core/
├── tools/
└── utils/
```

### 依存関係

```bash
pip install langchain langgraph openai pandas matplotlib
```

### 使用例

```python
from ai_analysis_engine import AIAnalysisEngine

engine = AIAnalysisEngine()
engine.initialize([...])
results = engine.analyze([...])
```
