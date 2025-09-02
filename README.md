# AI分析エンジン

AI分析エンジンは、アルゴリズムの評価結果を自動分析し、詳細な分析レポートを生成するシステムです。

## 概要

このシステムは以下の3つの主要コンポーネントで構成されています：

1. **Orchestrator** - 全体統制と最終レポート生成
2. **PerformanceAnalyzer** - 全体性能の確認・差分分析
3. **InstanceAnalyzer** - 個別データの詳細分析

## 主な特徴

- **自動期待値生成**: drowsy_detectionアルゴリズム仕様に基づき、評価結果から期待値を内部生成
- **DataWareHouse連携**: 評価結果の取得と分析結果の保存を自動化
- **包括的な分析**: 全体性能分析と個別データ分析の統合
- **可視化対応**: matplotlib/seaborn/plotlyを使用したグラフ生成
- **設定管理**: YAMLベースの柔軟な設定システム

## 技術仕様

- **Python**: 3.10+
- **主要ライブラリ**:
  - pandas: データ処理
  - pandasai: AI支援データ分析
  - langchain: LLM統合
  - matplotlib/seaborn/plotly: 可視化
  - jinja2: テンプレートエンジン
- **環境管理**: uv
- **データベース**: DataWareHouse API (SQLite3)
  - algorithm_api.py: アルゴリズム出力管理
  - analysis_api.py: 分析結果・問題点管理
  - evaluation_api.py: 評価データ管理

## インストール

```bash
# 仮想環境作成と依存関係インストール
uv venv
source .venv/Scripts/activate  # Windows
uv pip install -e .
```

## 使用方法

### 基本的な実行

```bash
# デフォルト設定で実行
python scripts/run_analysis.py

# 特定のアルゴリズム出力IDを指定
python scripts/run_analysis.py --algorithm-output-id 123

# カスタム設定ファイルを使用
python scripts/run_analysis.py --config config/production_config.yaml

# 詳細ログ出力
python scripts/run_analysis.py --verbose
```

### Pythonコードからの使用

```python
from ai_analysis_engine.orchestrator.orchestrator import Orchestrator
from ai_analysis_engine.config.settings import Settings

# 設定の読み込み
settings = Settings()

# Orchestratorの初期化
orchestrator = Orchestrator(settings)

# 分析実行
result = orchestrator.run_analysis()

# 結果の確認
if result['status'] == 'success':
    print(f"正解率: {result['integrated_results']['performance_summary']['accuracy']}")
    print(f"レポート: {result['report_path']}")
```

## 設定ファイル

### デフォルト設定 (config/default_config.yaml)

```yaml
global:
  database_path: ../DataWareHouse/database.db
  datawarehouse_path: ../DataWareHouse/
  templates_path: ./templates/

orchestrator:
  max_parallel_instances: 4
  timeout_seconds: 900

performance_analyzer:
  metrics:
    - accuracy
    - over_detection_count_per_hour
  visualization_level: standard

instance_analyzer:
  max_hypothesis_attempts: 3
  llm_model: gpt-4
  temperature: 0.1
  drowsy_detection:
    left_eye_close_threshold: 0.10
    right_eye_close_threshold: 0.10
    continuous_close_time: 1.00
    face_conf_threshold: 0.75
```

## プロジェクト構造

```
ai_analysis_engine/
├── src/ai_analysis_engine/
│   ├── orchestrator/          # Orchestratorクラス
│   ├── performance_analyzer/  # PerformanceAnalyzerクラス
│   ├── instance_analyzer/     # InstanceAnalyzerクラス
│   ├── utils/                 # ユーティリティ関数
│   └── config/                # 設定管理
├── config/                    # 設定ファイル
├── templates/                 # レポートテンプレート
├── tests/                     # テストコード
├── scripts/                   # 実行スクリプト
├── outputs/                   # 分析結果出力
│   ├── reports/              # 最終レポート
│   ├── charts/               # 図表ファイル
│   └── data/                 # 分析データ
└── logs/                     # 実行ログ
```

## 出力成果物

### 最終レポート (Markdown)
- 全体性能サマリー
- 個別データ分析結果
- 改善提案
- 可視化グラフへのリンク

### 図表ファイル (PNG)
- 時系列分析グラフ
- 混同行列
- 性能比較チャート

### 分析データ (JSON)
- 詳細な分析結果
- 指標データ
- 改善提案データ

## drowsy_detectionアルゴリズム仕様

本システムは、以下のアルゴリズム仕様に基づいて期待値を生成します：

- **入力**: frame_num, left_eye_open, right_eye_open, face_confidence
- **出力**: is_drowsy (1: 連続閉眼, 0: 非連続閉眼, -1: エラー)
- **パラメータ**:
  - 左目閉眼閾値: 0.10
  - 右目閉眼閾値: 0.10
  - 連続閉眼時間: 1.00秒
  - 顔検出信頼度閾値: 0.75

## DataWareHouse連携

- **データ取得**: `algorithm_output_table`から評価結果を取得
- **結果保存**: `05_analysis_output/`配下に分析結果を保存
- **データベース更新**: 分析結果を適切なテーブルに登録

## 開発・テスト

### テスト実行

```bash
# 単体テスト
python -m pytest tests/unit/ -v

# 統合テスト
python -m pytest tests/integration/ -v

# カバレッジレポート
python -m pytest --cov=src/ai_analysis_engine tests/
```

### 開発環境設定

```bash
# 開発依存関係のインストール
uv pip install -e .[dev]

# リンター実行
flake8 src/

# フォーマッター実行
black src/
```

## ライセンス

MIT License

## 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## 連絡先

プロジェクトに関する質問や提案は、Issueを作成してください。
